# Joint-COD-SOD (CVPR2021)
Uncertainty-aware Joint Salient Object and Camouflaged Object Detection
![alt text](./network_overview.png)

# Camouflage Maps
We provide camouflage maps on four camouflage testing datasets: CAMO, CHAMELEON, COD10K and the newly released NC4K from: "Simultaneously Localize, Segment and Rank the Camouflaged Objects":
https://drive.google.com/file/d/1FJikATesQCZJuu-wzKcWVFTaCL6rwVaj/view?usp=sharing

(Note that, as we retrain the model, the performance is slightly different from our reported number.)

```
@inproceedings{yunqiu_cod21,
  title={Simultaneously Localize, Segment and Rank the Camouflaged Objects},
  author={Lyu, Yunqiu and Zhang, Jing and Dai, Yuchao and Li, Aixuan and Liu, Bowen and Barnes, Nick and Fan, Deng-Ping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

# Code
Joint_learning: Joint camouflaged object detection and salient object detection model.

Joint_learning_sim: Joint camouflaged object detection and salient object detection model with similarity measure module.

Joint_learning_sim_adversarial: Joint camouflaged object detection and salient object detection model with similarity measure module and adversarial learning module (our final model).



# Our Bib:

Please cite our paper if necessary:
```
@inproceedings{aixuan_cod_sod21,
  title={Uncertainty-aware Joint Salient Object and Camouflaged Object Detection},
  author={Li, Aixuan and Zhang, Jing and Lyu, Yunqiu and Liu, Bowen and Zhang, Tong and Dai, Yuchao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021}
}
```

# Contact

Please drop me an email for further problems or discussion: zjnwpu@gmail.com
