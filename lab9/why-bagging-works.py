import numpy as np

samples = np.zeros((10,100))

for i in range(10):
    samples[i, :] = np.array(np.random.normal(i, 0.5, 100))

for i in range(10):
    print(np.var(samples[i]))

sum_samples = sum(samples)/len(samples)
print(np.var(sum_samples))


