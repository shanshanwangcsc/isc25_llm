# Running the Competition Tasks on CSC mahti server
## Clone the repository:
```bash
git clone https://github.com/phu0ngng/isc25_llm.git
cd isc25_llm
```
## change a few setting in the config.py

1. change the base cache dir
```bash
line 9: base_cache_dir: str = os.getenv('HF_HOME', os.path.expanduser("~/hf_cache"))
```
2. reduce the batch size to 4 so that you will not get out of memory error.
```bash
line 86: batch_size: int = 4 instead of 8
```
Even if you allocate 2 nodes (gpumedium), the batch size is set for per device as shown in src/trainer.py line 30,
```bash
per_device_train_batch_size=self.config.batch_size
```
 so keeping the batch size to 4.

3. For debugging, max steps is set to 1, for actual training, one needs to  change it to the actual training steps value
```bash
line 91: max_steps = 200 # set to 1 for debug
```
## changes in the src/trainer.py file
```bash
line 87: self.model to self.model.module
```
or else you  get the error like:  AttributeError: 'DistributedDataParallel' object has no attribute 'peft_config'.

## changes in the src/distributed.py file
```bash
line 23: return int(os.environ['LOCAL_RANK']) if dist.is_initialized() else 0
```

## change the run.sh to our run_mahti_1node.sh if you want to use 1 node, then submit the job in the terminal.

```bash
sbatch run_mahti_1node.sh
```

change the run.sh to our run_mahti_2nodes.sh if you want to use 2 nodes, then submit the job in the terminal.
```bash
sbatch run_mahti_2nodes.sh
```
