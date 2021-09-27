# -*- coding: cp1251 -*-
from devs import *
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from devs_dynamic_general import runDynamicGeneral
from devs_dynamic_shortest import runDynamicShortest
from devs_general import runGeneral
from devs_shortest import runShortest

maxAngents = 1000
arrivalRateMin = 0
arrivalRateMax = 3
service_xk = np.arange(8) + 1
service_pk = (0.05, 0.15, 0.1, 0.25 ,0.05, 0.2,0.1,0.1)
custm = stats.rv_discrete(name='custm', values=(service_xk, service_pk))

serviceTimes = [custm.rvs() for _ in range(2 * maxAngents)]
arrivalTimes = [random.randint(arrivalRateMin, arrivalRateMax) for _ in range(2 * maxAngents)]

res = pd.DataFrame([
        runGeneral(serviceTimes, arrivalTimes, maxAngents),
        runShortest(serviceTimes, arrivalTimes, maxAngents),
        runDynamicGeneral(serviceTimes, arrivalTimes, maxAngents),
        runDynamicShortest(serviceTimes, arrivalTimes, maxAngents)
    ], columns=[
        "Average waiting time",
        "Probability that a customer has to wait",
        "Probability of an Idle server",
        "Average service time",
        "Average time between arrivals",
        "Average waiting time for those who wait",
        "Average time a customer spends in the system",
        "Average time a customer spends in the system (alternative)"
])

pd.set_option("display.max_rows", None, "display.max_columns", None)

print("\n\n",res)


