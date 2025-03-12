
| data  | training loop | script_or_notebook| ddp support |usecase|
|-------| --------------|-------------------|-------------|-------|
|pytorch| manual        | notebook | No | benchmark_experiments/*ipynb|
|pytorch| manual        | script | Yes |benchmark_experiments/*_resnet.py|
| d2l   | d2l           | notebook | No |tiny_imagenet_baseline_toy.ipynb tiny_imagenet_baseline.ipynb|
| d2l   | d2l           | script | No ||

- Implement a manual data and trainer interface similar to d2l. Unit test the interface. 