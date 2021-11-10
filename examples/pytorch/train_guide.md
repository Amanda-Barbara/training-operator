# `pytorch`框架下训练`mnist`手写体识别流程

## `kubeflow`框架部署
* [`kubeflow`框架进行部署](https://github.com/Amanda-Barbara/kubeflow-manifests)
```shell
kind create cluster --config=kind/kind-config.yaml --name=kubeflow --image=kindest/node:v1.16.9
python install.py
kubectl get pod -nkubeflow
```

## `training-operator`镜像组件安装在`k8s`的命名空间`kubeflow`下面
```shell
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"
```

## 部署`PyTorchJob`训练服务
使用`PyTorchJob`训练PyTorch框架下的模型，在命名空间为`kubeflow`的`k8s`资源管理中验证是否支持`PyTorchJob`训练服务
```shell
kubectl get crd # 查看资源
```
如果显示如下，则说明功能组件`pytorchjobs.kubeflow.org`已安装，支持`PyTorchJob`训练服务
```text
NAME                                             CREATED AT
...
pytorchjobs.kubeflow.org                         2021-09-06T18:33:58Z
...
```
检查`Training operator`功能组件是否存在于命名空间为`kubeflow`的`k8s`资源管理中
```shell
kubectl get pods -n kubeflow
```
如果显示如下，则说明`Training operator`功能组件存在于命名空间为`kubeflow`的`k8s`资源管理中
```text
NAME                                READY   STATUS    RESTARTS   AGE
training-operator-d466b46bc-xbqvs   1/1     Running   0          4m37s
```
创建`PyTorch`训练工作
通过`PyTorchJob`创建一个训练任务
```shell
kubectl create -f https://raw.githubusercontent.com/kubeflow/training-operator/master/examples/pytorch/simple.yaml
```
查看被创建的`POD`单元
```shell
kubectl get pods -l job-name=pytorch-simple -n kubeflow
```
通过日志实时查看集群的训练任务
```shell
kubectl logs -f pod/pytorch-simple-master-0 -n kubeflow
```
日志如下：
```text
Using distributed PyTorch with gloo backend
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
Downloading http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
Processing...
Done!
2021-11-10T06:46:58Z INFO     Train Epoch: 1 [0/60000 (0%)]	loss=2.2975
2021-11-10T06:47:01Z INFO     Train Epoch: 1 [640/60000 (1%)]	loss=2.2965
2021-11-10T06:47:03Z INFO     Train Epoch: 1 [1280/60000 (2%)]	loss=2.2948
2021-11-10T06:47:06Z INFO     Train Epoch: 1 [1920/60000 (3%)]	loss=2.2833
2021-11-10T06:47:09Z INFO     Train Epoch: 1 [2560/60000 (4%)]	loss=2.2622
2021-11-10T06:47:12Z INFO     Train Epoch: 1 [3200/60000 (5%)]	loss=2.2193
2021-11-10T06:47:14Z INFO     Train Epoch: 1 [3840/60000 (6%)]	loss=2.2353
2021-11-10T06:47:17Z INFO     Train Epoch: 1 [4480/60000 (7%)]	loss=2.2295
2021-11-10T06:47:19Z INFO     Train Epoch: 1 [5120/60000 (9%)]	loss=2.1790
2021-11-10T06:47:22Z INFO     Train Epoch: 1 [5760/60000 (10%)]	loss=2.1150
2021-11-10T06:47:25Z INFO     Train Epoch: 1 [6400/60000 (11%)]	loss=2.0294
2021-11-10T06:47:27Z INFO     Train Epoch: 1 [7040/60000 (12%)]	loss=1.9156
2021-11-10T06:47:30Z INFO     Train Epoch: 1 [7680/60000 (13%)]	loss=1.7949
2021-11-10T06:47:33Z INFO     Train Epoch: 1 [8320/60000 (14%)]	loss=1.5567
2021-11-10T06:47:35Z INFO     Train Epoch: 1 [8960/60000 (15%)]	loss=1.3715
2021-11-10T06:47:38Z INFO     Train Epoch: 1 [9600/60000 (16%)]	loss=1.3386
2021-11-10T06:47:41Z INFO     Train Epoch: 1 [10240/60000 (17%)]	loss=1.1649
2021-11-10T06:47:44Z INFO     Train Epoch: 1 [10880/60000 (18%)]	loss=1.0924
2021-11-10T06:47:46Z INFO     Train Epoch: 1 [11520/60000 (19%)]	loss=1.0665
2021-11-10T06:47:49Z INFO     Train Epoch: 1 [12160/60000 (20%)]	loss=1.0488
2021-11-10T06:47:51Z INFO     Train Epoch: 1 [12800/60000 (21%)]	loss=1.3654
2021-11-10T06:47:54Z INFO     Train Epoch: 1 [13440/60000 (22%)]	loss=1.0043
2021-11-10T06:47:57Z INFO     Train Epoch: 1 [14080/60000 (23%)]	loss=0.9411
2021-11-10T06:47:59Z INFO     Train Epoch: 1 [14720/60000 (25%)]	loss=0.8942
2021-11-10T06:48:02Z INFO     Train Epoch: 1 [15360/60000 (26%)]	loss=0.9586
2021-11-10T06:48:05Z INFO     Train Epoch: 1 [16000/60000 (27%)]	loss=1.1150
2021-11-10T06:48:07Z INFO     Train Epoch: 1 [16640/60000 (28%)]	loss=1.0944
2021-11-10T06:48:10Z INFO     Train Epoch: 1 [17280/60000 (29%)]	loss=0.8610
2021-11-10T06:48:13Z INFO     Train Epoch: 1 [17920/60000 (30%)]	loss=0.9365
2021-11-10T06:48:16Z INFO     Train Epoch: 1 [18560/60000 (31%)]	loss=0.7595
2021-11-10T06:48:18Z INFO     Train Epoch: 1 [19200/60000 (32%)]	loss=0.8755
2021-11-10T06:48:21Z INFO     Train Epoch: 1 [19840/60000 (33%)]	loss=1.1830
2021-11-10T06:48:23Z INFO     Train Epoch: 1 [20480/60000 (34%)]	loss=0.7637
2021-11-10T06:48:26Z INFO     Train Epoch: 1 [21120/60000 (35%)]	loss=0.8971
2021-11-10T06:48:29Z INFO     Train Epoch: 1 [21760/60000 (36%)]	loss=0.7019
2021-11-10T06:48:32Z INFO     Train Epoch: 1 [22400/60000 (37%)]	loss=0.7468
2021-11-10T06:48:35Z INFO     Train Epoch: 1 [23040/60000 (38%)]	loss=0.8303
2021-11-10T06:48:38Z INFO     Train Epoch: 1 [23680/60000 (39%)]	loss=0.8403
2021-11-10T06:48:40Z INFO     Train Epoch: 1 [24320/60000 (41%)]	loss=0.8833
2021-11-10T06:48:43Z INFO     Train Epoch: 1 [24960/60000 (42%)]	loss=0.8821
2021-11-10T06:48:46Z INFO     Train Epoch: 1 [25600/60000 (43%)]	loss=0.6553
2021-11-10T06:48:48Z INFO     Train Epoch: 1 [26240/60000 (44%)]	loss=0.8553
2021-11-10T06:48:51Z INFO     Train Epoch: 1 [26880/60000 (45%)]	loss=0.8560
2021-11-10T06:48:54Z INFO     Train Epoch: 1 [27520/60000 (46%)]	loss=0.9439
2021-11-10T06:48:56Z INFO     Train Epoch: 1 [28160/60000 (47%)]	loss=0.7415
2021-11-10T06:48:59Z INFO     Train Epoch: 1 [28800/60000 (48%)]	loss=0.8245
2021-11-10T06:49:02Z INFO     Train Epoch: 1 [29440/60000 (49%)]	loss=0.8443
2021-11-10T06:49:05Z INFO     Train Epoch: 1 [30080/60000 (50%)]	loss=0.6781
2021-11-10T06:49:07Z INFO     Train Epoch: 1 [30720/60000 (51%)]	loss=0.9853
2021-11-10T06:49:10Z INFO     Train Epoch: 1 [31360/60000 (52%)]	loss=0.8705
2021-11-10T06:49:13Z INFO     Train Epoch: 1 [32000/60000 (53%)]	loss=0.6735
2021-11-10T06:49:16Z INFO     Train Epoch: 1 [32640/60000 (54%)]	loss=0.7951
2021-11-10T06:49:19Z INFO     Train Epoch: 1 [33280/60000 (55%)]	loss=0.8220
2021-11-10T06:49:21Z INFO     Train Epoch: 1 [33920/60000 (57%)]	loss=0.8706
2021-11-10T06:49:24Z INFO     Train Epoch: 1 [34560/60000 (58%)]	loss=0.9538
2021-11-10T06:49:27Z INFO     Train Epoch: 1 [35200/60000 (59%)]	loss=0.6991
2021-11-10T06:49:29Z INFO     Train Epoch: 1 [35840/60000 (60%)]	loss=0.7417
2021-11-10T06:49:32Z INFO     Train Epoch: 1 [36480/60000 (61%)]	loss=0.8806
2021-11-10T06:49:35Z INFO     Train Epoch: 1 [37120/60000 (62%)]	loss=0.5654
2021-11-10T06:49:38Z INFO     Train Epoch: 1 [37760/60000 (63%)]	loss=0.8553
2021-11-10T06:49:40Z INFO     Train Epoch: 1 [38400/60000 (64%)]	loss=0.6486
2021-11-10T06:49:43Z INFO     Train Epoch: 1 [39040/60000 (65%)]	loss=0.5933
2021-11-10T06:49:46Z INFO     Train Epoch: 1 [39680/60000 (66%)]	loss=0.5394
2021-11-10T06:49:48Z INFO     Train Epoch: 1 [40320/60000 (67%)]	loss=0.7578
2021-11-10T06:49:51Z INFO     Train Epoch: 1 [40960/60000 (68%)]	loss=0.5938
2021-11-10T06:49:54Z INFO     Train Epoch: 1 [41600/60000 (69%)]	loss=0.7355
2021-11-10T06:49:55Z INFO     Train Epoch: 1 [42240/60000 (70%)]	loss=0.7312
2021-11-10T06:49:56Z INFO     Train Epoch: 1 [42880/60000 (71%)]	loss=0.7593
2021-11-10T06:49:59Z INFO     Train Epoch: 1 [43520/60000 (72%)]	loss=0.7412
2021-11-10T06:50:02Z INFO     Train Epoch: 1 [44160/60000 (74%)]	loss=0.5995
2021-11-10T06:50:05Z INFO     Train Epoch: 1 [44800/60000 (75%)]	loss=0.6418
2021-11-10T06:50:07Z INFO     Train Epoch: 1 [45440/60000 (76%)]	loss=0.8501
2021-11-10T06:50:10Z INFO     Train Epoch: 1 [46080/60000 (77%)]	loss=0.8012
2021-11-10T06:50:13Z INFO     Train Epoch: 1 [46720/60000 (78%)]	loss=0.9049
2021-11-10T06:50:16Z INFO     Train Epoch: 1 [47360/60000 (79%)]	loss=0.5929
2021-11-10T06:50:18Z INFO     Train Epoch: 1 [48000/60000 (80%)]	loss=0.5918
2021-11-10T06:50:21Z INFO     Train Epoch: 1 [48640/60000 (81%)]	loss=0.6389
2021-11-10T06:50:24Z INFO     Train Epoch: 1 [49280/60000 (82%)]	loss=0.5233
2021-11-10T06:50:26Z INFO     Train Epoch: 1 [49920/60000 (83%)]	loss=0.9672
2021-11-10T06:50:29Z INFO     Train Epoch: 1 [50560/60000 (84%)]	loss=0.7550
2021-11-10T06:50:32Z INFO     Train Epoch: 1 [51200/60000 (85%)]	loss=0.6280
2021-11-10T06:50:35Z INFO     Train Epoch: 1 [51840/60000 (86%)]	loss=0.5377
2021-11-10T06:50:37Z INFO     Train Epoch: 1 [52480/60000 (87%)]	loss=0.6016
2021-11-10T06:50:40Z INFO     Train Epoch: 1 [53120/60000 (88%)]	loss=0.4454
2021-11-10T06:50:43Z INFO     Train Epoch: 1 [53760/60000 (90%)]	loss=0.7935
2021-11-10T06:50:45Z INFO     Train Epoch: 1 [54400/60000 (91%)]	loss=0.5740
2021-11-10T06:50:48Z INFO     Train Epoch: 1 [55040/60000 (92%)]	loss=0.6581
2021-11-10T06:50:51Z INFO     Train Epoch: 1 [55680/60000 (93%)]	loss=0.5466
2021-11-10T06:50:54Z INFO     Train Epoch: 1 [56320/60000 (94%)]	loss=0.5859
2021-11-10T06:50:57Z INFO     Train Epoch: 1 [56960/60000 (95%)]	loss=0.5472
2021-11-10T06:50:59Z INFO     Train Epoch: 1 [57600/60000 (96%)]	loss=0.7145
2021-11-10T06:51:02Z INFO     Train Epoch: 1 [58240/60000 (97%)]	loss=0.7311
2021-11-10T06:51:05Z INFO     Train Epoch: 1 [58880/60000 (98%)]	loss=0.8890
2021-11-10T06:51:07Z INFO     Train Epoch: 1 [59520/60000 (99%)]	loss=0.5368
2021-11-10T06:51:12Z INFO     {metricName: accuracy, metricValue: 0.7313};{metricName: loss, metricValue: 0.6649}
```


## 参考链接
* 1 [`pytorch`框架下训练`mnist`手写体识别流程](https://www.kubeflow.org/docs/components/training/pytorch/)
* 2 [`training-operator`镜像安装](https://github.com/Amanda-Barbara/training-operator)




