from pypsupertime import Psupertime

p=Psupertime(n_batches=2, n_jobs=1, estimator_params={"learning_rate": "optimal", "early_stopping_batches":True, "penalty": "l1"})

p.run("/home/julian/Uni/MasterThesis/data/acinar_sce.h5ad", "donor_age")



