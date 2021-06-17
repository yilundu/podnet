You may use the following command to train the PODNet model

First train the model utilizing only MONET loss

```python
python train.py --name intphys_physics_test --model monet --dataroot .  --batch_size 16
```

Then train the model with physics loss

```python
python train.py --name intphys_physics_test --continue_train --physics_loss --model monet --dataroot .  --batch_size 16
```

You may then eval a model with the command (pretrain model is attached)

```python
python eval.py --name intphys_physics_bce_new --model monet --dataroot . --batch_size 5 --continue_train --eval_intphys
```

