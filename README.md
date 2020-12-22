# How to run repaired nets:

### These scripts can be used to evaluate the repaired badnets. It uses spectral signatures from poisoned images to detect the poisoning. It is able to capture 99% of the posined images if not all, with extremely low false positive rates of < 1%.

- Install all packages in requirements.txt in the environment you are running the code.
- The repaired nets scripts are name badnetname_repaired.py
- To get the prediction on a test set, simply run python ./badnetname_repaired.py `path/to/testimage`
- The prediction outputs are printed on the screen