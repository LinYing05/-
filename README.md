本项目仅用于课程学习。
This experimental project is for course study only.

### 许可声明

本项目部分代码和模型基于yeephycho的tensorflow-face-detection项目（https://github.com/yeephycho/tensorflow-face-detection），该项目遵循Apache 2.0许可。以下是Apache 2.0许可的全文：

Copyright 2017 - 2020 yeephycho

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

本项目依赖于Google TensorFlow对象检测API。TensorFlow遵循其自身的许可协议，请参考（https://github.com/yeephycho/tensorflow-face-detection/blob/master/LICENSE）获取详细信息。

本项目使用了WIDERFACE数据集进行模型训练。WIDERFACE数据集遵循其自身的许可协议，请参考http://shuoyang1213.me/WIDERFACE/获取详细信息。

### 运行过程

1.	将Host操作系统端(Windows11)摄像头同步至Guest虚拟机端操作系统(ubuntu16.04)，步骤如下：  
    a)	在VirtualBox中启动Guest操作系统(Ubuntu-16.04)。  
    b)	在host操作系统(Windows11)下打开cmd，用cd命令进入VirtualBox所安装的地方，也可以手动进入目录，输入cmd直接进入。  
    c)	之后用VBoxManage list webcams命令列出所有存在的摄像头。  
  	d)	设备只有一个USB连接驱动的摄像头，这时接着运行如下命令将第一个摄像头同步到虚拟机中的Ubuntu操作系统：  
        VboxManage controlvm "Ubuntu-16.04" webcam attach .1  
    e)	之后，可以在VirtualBox中点级“设备”->“摄像头”看到同步过来的摄像头，单机打勾。  
  	f)	在Ubuntu中验证摄像头是否正常工作：  
            i.	在Ubuntu中打开Terminal并运行sodo apt-get install cheese  
  	        ii.	之后在terminal中直接运行cheese，可以看到摄像影像，则说明同步成功：  
2.	在Ubuntu中安装git，在terminal中运行：sudo apt-get install git
3.	本次运行的项目是基于MobileNet的人脸检测项目，github链接是https://github.com/yeephycho/tensorflow-face-detection
4.	在Ubuntu的Terminal中运行  
    git clone https://github.com/yeephycho/tensorflow-face-detection.git  
5.	在Ubuntu的Terminal用cd命令进入第4步下载好的tensorflow-face-detection-master项目里面，并运行如下指令：  
    python3.8 inference_usbCam_face.py 0  
    出现人脸识别结果则成功。
