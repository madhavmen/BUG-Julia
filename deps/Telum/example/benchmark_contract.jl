using Revise
using LurCGT
using Telum
using LinearAlgebra

include("../test/support/dmrg.jl")

function benchmark_smallRMT()
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q = getLocalSpace(option);
    qi1 = TLArray(q.I, ("lur1", "lur1"))
    qi2 = TLArray(q.I, ("lur2", "lur2"))
    qf = TLArray(q.F, ("lur2", "lur2", "op"))
    a = getIdentity((qi1, 2), (qi2, 2));
    return qf, a
end

do_Lanczos(Hl, Hr, M, H1, H2) = eigs_GS(Hl, [H1, H2], Hr, M; tol=1e-8, nKrylov=5, time_blocks=true)

function benchmark_DMRGres(MPS, MPO, Hrl)
    Hl = Hrl[21]
    Hr = Hrl[23]'
    M = MPS[21] * MPS[22]
    H1, H2 = MPO[21], MPO[22]

    return Hl, Hr, M, H1, H2
end

function get_DMRGres(Nkeep=50)
    MPO = HubbardMPO(4.0, 1.5, 1.0, 40)
    MPS, E, sp = init_MPS(MPO, Nkeep; tol=0.0)
    DMRG_GS_2site!(MPS, MPO, Nkeep, 1)
    Hrl = getHrl(MPO, MPS)

    return MPS, MPO, Hrl
end

function benchmark_sum(vec)
    @time for _=1:100 sum(vec) end
end

function svd_testtensor(Nkeep=50)
    MPS, MPO, Hrl = get_DMRGres(Nkeep)
    Hl, Hr, M, H1, H2 = benchmark_DMRGres(MPS, MPO, Hrl)
    return do_Lanczos(Hl, Hr, M, H1, H2)
end
