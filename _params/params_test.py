# %%
import sys
sys.path.insert(0, './_params')
local = True
if local:
    import params_local
    params = params_local
elif local == False:
    import params_Niagara
    params = params_Niagara
# %%
n_x = int(params.n_x)
n_z = int(params.n_z)
print(n_x, n_z)
# %%
foo = 0
bar = bool(foo)
print(bar)
