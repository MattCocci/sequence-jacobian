import Main


## Initialize
runcode = 1
T       = 300
Npd     = 300  # Number of periods to simulate
Nsim    = 2
fromMA  = True

hempirical = False
hfromMA    = False
seed       = 314

resim     = False
rematch   = True
nparallel = 2


# Simulate and Save
if resim:
    Main.muhatsSimSave(runcode, T, Npd, Nsim, fromMA, save=True, verbose=False)


if rematch:
    res = Main.MomentMatchRuns(runcode, T, Npd, Nsim, fromMA, hempirical, hfromMA, nparallel)


