# *Optimal Prompt Guided Multimodal Learning for Blind Omnidirectional Image Quality Assessment*
Jiebin Yan<sup>1</sup>, Jiayu Zhang<sup>1</sup>, Jiale Rao<sup>1</sup>, Lei Wu<sup>1</sup>, Pengfei Cheng<sup>2</sup>, and Yuming Fang<sup>1</sup>.

<sup>1</sup> School of Computing and Artificial Intelligence, Jiangxi University of Finance and Economics

<sup>2</sup> School of Artificial Intelligence,  Xidian University

## :four_leaf_clover:Generate quality-aware description

```python
CUDA_VISIBLE_DEVICES=0 python data_instruct.py --directory '/mnt/10T/wkc/Database/0IQ-10K/0IQ-10K_image' --csv_path '/mnt/10T/zjy/database/oiq_10k.csv'  --text_prompt 'Your prompt' --save_folder_path 'Your save_folder_path' --csv_folder_path 'Your csv_folder_path' --csv_name 'OIQ-10K' --save_name_path 'OIQ-10K'
``` 



## :dart:OPML Architecture
<p align="center"><img src="https://github.com/jiayuzhangj/OPML/blob/main/image/OPML.png" width="900"></p>

The framework of the proposed model. It consists of three parts: (a) optimal prompt selection module, (b) hierarchical visual degradation modeling branch, and (c) multimodal quality capturing module. To better and fully capture the detailed information of OIs and further enhance the fusion of multimodal data, the DME module is used to attenuate the degradation information, while the MQC module is used to enhance the multimodal feature representation.



### Train and Test

Edit `config.py` for configuration

* Train

```python
CUDA_VISIBLE_DEVICES=0 python train.py

