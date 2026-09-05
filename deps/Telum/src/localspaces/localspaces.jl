# Fermionic annihilation operator
const f2 = [0 1; 0 0]
# Fermionic parity operator
const z2 = [1 0; 0 -1]
# Fermionic identity operator
const I2 = [1 0; 0 1]

const spin2 = [1 0; 0 -1]

const f4up = f2 ⊗ I2
const f4down = z2 ⊗ f2
const z4 = f4up' * f4up + f4down' * f4down
# Spin z-operator for spinful fermion
const sz4 = f4up' * f4up - f4down' * f4down

include("options.jl")
include("Spin.jl")
include("Fermion.jl")
include("FermionS.jl")
