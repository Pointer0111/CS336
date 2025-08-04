from torch.nn.modules import padding
import triton
import triton.language as tl
import torch

class PytorchFLA2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        # 兼容3D和4D张量
        Nq, d = Q.shape[-2:]
        Nk = K.shape[-2]
        B = Q.shape[0]
        Bq, Bk = 16, 16
        Tq = triton.cdiv(Nq, Bq)
        Tk = triton.cdiv(Nk, Bk)

        O = torch.zeros_like(Q)
        L = torch.zeros(Q.shape[0], Nq, device=Q.device)  # [B, Nq]

        for i in range(Tq):
            qs = i * Bq
            qe = min((i + 1) * Bq, Nq)
            Qi = Q[:, qs:qe, :]  # [B, Bq, d]
            Oi = torch.zeros(Q.shape[0], qe - qs, d, device=Q.device)  # [B, Bq, d]
            mi = torch.full((Q.shape[0], qe - qs), float('-inf'), device=Q.device)  # [B, Bq]
            li = torch.zeros(Q.shape[0], qe - qs, device=Q.device)  # [B, Bq]
            for j in range(Tk):
                ks = j * Bk
                ke = min((j + 1) * Bk, Nk)
                Kj = K[:, ks:ke, :]  # [B, Bk, d]
                Vj = V[:, ks:ke, :]  # [B, Bk, d]
                # 计算注意力分数
                Sij = torch.einsum('bid,bjd->bij', Qi, Kj) / (d ** 0.5)  # [B, Bq, Bk]
                mj = Sij.max(dim=2).values  # [B, Bq]
                m_new = torch.maximum(mi, mj)  # [B, Bq]
                exp_mi_mnew = torch.exp(mi - m_new)  # [B, Bq]
                exp_Sij_mnew = torch.exp(Sij - m_new.unsqueeze(-1))  # [B, Bq, Bk]
                pij = exp_Sij_mnew  # [B, Bq, Bk]
                lij = pij.sum(dim=2)  # [B, Bq]
                # 累加
                Oi = Oi * exp_mi_mnew.unsqueeze(-1) + torch.einsum('bij,bjd->bid', pij, Vj)
                li = li * exp_mi_mnew + lij
                mi = m_new
            O[:, qs:qe, :] = Oi / li.unsqueeze(-1)
            L[:, qs:qe] = mi + torch.log(li)
            
        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O


    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        # Q: [B, Nq, d], K: [B, Nk, d], V: [B, Nk, d], O: [B, Nq, d], dO: [B, Nq, d], L: [B, Nq]
        B, Nq, d = Q.shape
        Nk = K.shape[1]
        Bq, Bk = 16, 16
        Tq = triton.cdiv(Nq, Bq)
        Tk = triton.cdiv(Nk, Bk)

        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)

        D = torch.sum(dO * O, dim=-1)  # [B, Nq]

        for j in range(Tk):
            ks = j * Bk
            ke = min((j + 1) * Bk, Nk)
            K_j = K[:, ks:ke, :]  # [B, Bk, d]
            V_j = V[:, ks:ke, :]  # [B, Bk, d]
            dK_j = torch.zeros_like(K_j)
            dV_j = torch.zeros_like(V_j)
            for i in range(Tq):
                qs = i * Bq
                qe = min((i + 1) * Bq, Nq)
                Qi = Q[:, qs:qe, :]  # [B, Bq, d]
                Oi = O[:, qs:qe, :]  # [B, Bq, d]
                dOi = dO[:, qs:qe, :]  # [B, Bq, d]
                Li = L[:, qs:qe]  # [B, Bq]
                Di = D[:, qs:qe]  # [B, Bq]
                # Sij: [B, Bq, Bk]
                Sij = torch.einsum('bid,bjd->bij', Qi, K_j) / (d ** 0.5)
                Pij = torch.exp(Sij - Li.unsqueeze(-1))  # [B, Bq, Bk]
                # dV_j += Pij^T @ dOi
                dV_j = dV_j + torch.einsum('bij,bid->bjd', Pij, dOi)
                # dPij = dOi @ V_j^T
                dPij = torch.einsum('bid,bjd->bij', dOi, V_j)
                # ds = Pij * (dPij - Di.unsqueeze(-1)) / sqrt(d)
                ds = Pij * (dPij - Di.unsqueeze(-1)) / (d ** 0.5)
                # dQi = ds @ K_j
                dQi = torch.einsum('bij,bjd->bid', ds, K_j)
                dQ[:, qs:qe, :] += dQi
                # 用matmul替换einsum
                # ds: [B, Bq, Bk], Qi: [B, Bq, d] -> ds.transpose(1,2): [B, Bk, Bq]
                dK_j = dK_j + torch.matmul(ds.transpose(1, 2), Qi)  # [B, Bk, d]
            dK[:, ks:ke, :] += dK_j
            dV[:, ks:ke, :] += dV_j
        return dQ, dK, dV, None


        
class TritonFLA2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        Nq, d = Q.shape[-2:]
        Nk = K.shape[-2]
        Bq, Bk = 16, 16
        ctx.Bq, ctx.Bk = Bq, Bk
        ctx.is_causal = is_causal
        Tq = triton.cdiv(Nq, Bq)
        B = Q.shape[0]

        scale = 1 / (d ** 0.5)
        O = torch.zeros_like(Q)
        L = torch.zeros(B, Nq, device=Q.device)

        # 不能用变量传递 constexpr，必须用常量
        if is_causal:
            flash_fwd_kernel[(Tq, B,)](
                Q, K, V,
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                Nq, Nk,
                scale,
                D=d,
                Q_TILE_SIZE=16,
                K_TILE_SIZE=16,
                is_causal=True,
            )
        else:
            flash_fwd_kernel[(Tq, B,)](
                Q, K, V,
                O, L,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K.stride(0), K.stride(1), K.stride(2),
                V.stride(0), V.stride(1), V.stride(2),
                O.stride(0), O.stride(1), O.stride(2),
                L.stride(0), L.stride(1),
                Nq, Nk,
                scale,
                D=d,
                Q_TILE_SIZE=16,
                K_TILE_SIZE=16,
                is_causal=False,
            )

        ctx.save_for_backward(Q, K, V, O, L)
        return O

    @staticmethod
    def backward(ctx, grad_output):
        Q, K, V, O, L = ctx.saved_tensors
        B, Nq, d = Q.shape
        Nk = K.shape[1]
        Bq, Bk = 16, 16
        is_causal = ctx.is_causal
        Tq = triton.cdiv(Nq, Bq)
        Tk = triton.cdiv(Nk, Bk)
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        scale = 1 / (d ** 0.5)

        # 先计算 dQ
        flash_bwd_dq_kernel[(Tq, B)](
            Q, K, V, O, grad_output, L,
            dQ,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=16,
            K_TILE_SIZE=16,
            is_causal=is_causal,
        )
        # 再计算 dK, dV
        flash_bwd_dk_dv_kernel[(Tk, B)](
            Q, K, V, O, grad_output, L,
            dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            Nq, Nk,
            scale,
            D=d,
            Q_TILE_SIZE=16,
            K_TILE_SIZE=16,
            is_causal=is_causal,
        )
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    # 程序索引
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    # 创建块指针
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )

    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )

    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")

    O_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    mi = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    li = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for i in range(Tk):
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Sij = tl.dot(Q_block, tl.trans(K_block)) * scale
        # 实现 causal mask
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = i * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] < k_idx[None, :]
            Sij = tl.where(mask, Sij - 1e6, Sij)
        mj = tl.max(Sij, axis=1)
        m_new = tl.maximum(mi, mj)
        exp_mi_mnew = tl.exp(mi - m_new)
        exp_Sij_mnew = tl.exp(Sij - m_new[:, None])
        pij = exp_Sij_mnew
        lij = tl.sum(pij, axis=1)
        # O_block严格按论文公式归并
        O_block = O_block * exp_mi_mnew[:, None] + tl.dot(pij, V_block)
        li = li * exp_mi_mnew + lij
        mi = m_new
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    # 归一化输出
    tl.store(O_block_ptr, O_block / li[:, None], boundary_check=(0, 1))
    tl.store(L_block_ptr, mi + tl.log(li), boundary_check=(0,))


# dQ kernel: 以 Q-tile 为主循环
@triton.jit
def flash_bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dQ_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dQ_block_ptr = tl.make_block_ptr(
        dQ_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    O_block = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
    dO_block = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
    L_block = tl.load(L_block_ptr, boundary_check=(0,))
    dQ_block = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    D_block = tl.sum(dO_block * O_block, axis=1)  # [Q_TILE_SIZE]
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    # 只创建一次K_block_ptr和V_block_ptr
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    for j in range(Tk):
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Sij = tl.dot(Q_block, tl.trans(K_block)) * scale
        if is_causal:
            q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] < k_idx[None, :]
            Sij = tl.where(mask, Sij - 1e6, Sij)
        Pij = tl.exp(Sij - L_block[:, None])
        dP = tl.dot(dO_block, tl.trans(V_block))
        dS = Pij * (dP - D_block[:, None]) * scale
        dQ_block += tl.dot(dS, K_block)
        # advance block ptrs
        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))
    tl.store(dQ_block_ptr, dQ_block, boundary_check=(0, 1))

# dK/dV kernel: 以 K-tile 为主循环
@triton.jit
def flash_bwd_dk_dv_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr, dO_ptr, L_ptr,
    dK_ptr, dV_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    k_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block_ptr = tl.make_block_ptr(
        dK_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dV_block_ptr = tl.make_block_ptr(
        dV_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(k_tile_index * K_TILE_SIZE, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    dK_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    dV_block = tl.zeros((K_TILE_SIZE, D), dtype=tl.float32)
    Tq = tl.cdiv(N_QUERIES, Q_TILE_SIZE)
    # 只创建一次Q_block_ptr等
    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    dO_block_ptr = tl.make_block_ptr(
        dO_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(0, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(0,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )
    for qi in range(Tq):
        Q_block = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
        O_block = tl.load(O_block_ptr, boundary_check=(0, 1), padding_option="zero")
        dO_block = tl.load(dO_block_ptr, boundary_check=(0, 1), padding_option="zero")
        L_block = tl.load(L_block_ptr, boundary_check=(0,))
        D_block = tl.sum(dO_block * O_block, axis=1)  # [Q_TILE_SIZE]
        K_block = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        V_block = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Sij = tl.dot(Q_block, tl.trans(K_block)) * scale
        if is_causal:
            q_idx = qi * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            k_idx = k_tile_index * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = q_idx[:, None] < k_idx[None, :]
            Sij = tl.where(mask, Sij - 1e6, Sij)
        Pij = tl.exp(Sij - L_block[:, None])
        dV_block += tl.dot(tl.trans(Pij), dO_block)
        dP = tl.dot(dO_block, tl.trans(V_block))
        dS = Pij * (dP - D_block[:, None]) * scale
        dK_block += tl.dot(tl.trans(dS), Q_block)
        # advance block ptrs
        Q_block_ptr = Q_block_ptr.advance((Q_TILE_SIZE, 0))
        O_block_ptr = O_block_ptr.advance((Q_TILE_SIZE, 0))
        dO_block_ptr = dO_block_ptr.advance((Q_TILE_SIZE, 0))
        L_block_ptr = L_block_ptr.advance((Q_TILE_SIZE,))
    tl.store(dK_block_ptr, dK_block, boundary_check=(0, 1))
    tl.store(dV_block_ptr, dV_block, boundary_check=(0, 1))





