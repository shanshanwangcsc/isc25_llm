# Running the Competition Tasks on CSC mahti server
## Clone the repository:
```bash
git clone https://github.com/phu0ngng/isc25_llm.git
cd isc25_llm
```
## change a few setting in the config.py
a. reduce the batch size to 4 so that you will not get out of memory error.
```bash
line 86, batch_size: int = 4 instead of 8
```
Even if you allocate 2 nodes (gpumedium), the batch size is set for per device as shown in src/trainer.py line 30,
```bash
per_device_train_batch_size=self.config.batch_size
```
 so keeping the batch size to 4.

b. For debugging, max steps is set to 1, for actual training, one needs to  change it to the actual training steps value
```bash
line 91, max_steps = 12 # set to 1 for debug
```
## changes in the src/trainer.py file
```bash
line 87,self.model to self.model.module
```
or else you  get the error like:  AttributeError: 'DistributedDataParallel' object has no attribute 'peft_config'.

## change the run.sh to our run_mahti.sh, then submit the job in the terminal.

```bash
sbatch run_mahti.sh
```
