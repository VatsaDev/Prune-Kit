# Making your model more efficient (Prune-Kit)

Features:

 - pruning layers
 - pruning KV heads
 - SFT recovery/perf gain
 - Full model distillation
 - Knocking out layers and testing performance
 - Lora

## How to use

#### step 0: collect everything

get the `teacher_model`, `model_you_want_to_prune`, `train.json`, `val.json`, all in the server

init steps for the libs are the same as any other ms-swift step, I used UV for speed. 

#### step 1: testing low hanging fruit

add datasets and checkpoints to `knockout_check.sh`, add the layers from text and vision you would like to remove, let it cook, and look at wandb

the loss will look something like this: <br><br>
<img width="529" height="437" alt="image" src="https://github.com/user-attachments/assets/5e840592-09e1-4696-8fca-56f18e55cf5a" />
> anything with a decent spike at the beginning is a key layer and will most likely give you a permanent 5-10% decrease in performance

the best method to filter for pruning finalists is `eval/token_acc` <br><br>

<img width="473" height="428" alt="image" src="https://github.com/user-attachments/assets/c1d64389-6fc7-40bb-be9b-bab2a9294cb6" />

anything over a 95% is basically free to remove, anything over 90% is probably fine after recovery, dont do more than like 10-25% of the models weights or over 10 layers at once (the earlier you are in pruning, the more layers you can remove), that tends to end badly, gradually recovering and removing them again has worked best for me

check eval token acc, collect a final list of what text/vision you want to prune

#### step 2: pruning and recovery

go to `manual_select.py` and edit that with your list to remove, should get an output like this: <br><br>

<img width="316" height="90" alt="image" src="https://github.com/user-attachments/assets/e7a85f14-7cb9-4f34-97f1-6ba2b6713504" />

take those new weights, replace `MODEL_DIR` in `full_run.sh`, run the recovery, aim for around 95-96% eval/token_acc for prod

heres a recovery run chart: <br><br>
<img width="780" height="574" alt="image" src="https://github.com/user-attachments/assets/6b0d83be-7758-44ec-b6a8-951abf271b12" />

take this out model to step 3

#### step 3: distillation

get your strong teacher model (in my case, a strong output 8b_RL model), replace the path for student model in `distil.sh`, run it

the strong models in my runs got to `train/eval` loss of `0.2/0.8` 

<img width="1829" height="368" alt="image" src="https://github.com/user-attachments/assets/37986fa9-b206-481d-94d0-81755457aa0e" />

> generally any healthy run wont start over 0.8 loss, something might be wrong if it is 

<br><br><br>

repeat steps 1-3 until at your desired model size and performance

#### step 4, chop the KV heads off

go to `KV_cut.py`, put your model in, set the number of heads you want (I recommend starting at `old_kv//2`, or `4`, gradually push to `1`)

the KV runs will match the rest of the recovery SFT, same principle, I rec you keeping going till you have MQA (KV=1), most models are like that now, and compared to default 8 KV for Qwen, 8x gain on mem to 1 KV head is nice.

#### step 5, LoRA (not nesc)

if the model is lacking something small (ex. my case was poor multilingual support), use the lora script with an exclusive dataset, currently it tunes 4M params (0.4% of the model), but it improved performance without degradation. change rank/alpha as nesc.

lora FT at `lora.sh` -> produces an adapter, we dont really want the trouble of serving that at inference time, so export back to a full normal model by editing `export_lora.sh`, it does use absolute paths because thats how the adapter linked to orig_ckpt, its buggy otherwise

#### step 5, RL 

All the GRPO run settings to improve the model are inside `rl_run.sh` along with all rewards I test under `plugins`

Generally a good RL run had a slow but steady upward drift, pushing near the middle, I recommend using `GSPO`

<img width="2409" height="923" alt="image" src="https://github.com/user-attachments/assets/b4ec7c7a-c4f5-48a1-9c77-1233d652f72d" />
