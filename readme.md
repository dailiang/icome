# 说明
这是 2013 年** "[中国云-移动互联网创新大赛](http://2013.icome.org.cn/)" **的** "[技术类赛题 2](http://2013.icome.org.cn/subjects/10#002) ：慧眼识人——轻松捕捉剪影—— 勾勒人物图片中的人物轮廓" **, 此算法采用深度的卷积神经网络对图片进行分割,详细说明见另外两个 pdf 文档.

##更新(结果已出)
比赛结果已经公布, 前十名中有两支队伍采用了深度学习算法. 另外一支果然是来自中科院的团队, 最终夺得第一名,恭喜啦~    
好吧, 我的得分比较靠后, 参数难调, GPU 太慢. 再接再厉吧, 详情请见百度开放社区公布的** [leaderboard](http://openresearch.baidu.com/activityprogress.jhtml?channelId=576) **, 哈哈, 倒数第二就是我啦.


##1. 团队基本信息    
团队名称：  DeepLearner    
团队编号：  309号团队
领队姓名：  DanielD (实际上就我一个人, - - )    
领队邮件地址：  986639399@qq.com     

##2. 运行环境    
操作系统：Ubuntu 12.04，64bit，内核 Linux 3.5.0    
编程语言：python 2.7.3    
版本说明：    
numpy-1.6.1    
scipy-0.9.0 (此版本会出现一个warning，但不影响运行，0.12以上版本修复了这个bug）    
PIL-1.1.7 (64位系统默认的 PIL 没有安装jpeg和png的库的,需要单独安装）    
theano-0.6.0rc3    


##3. 运行命令

$  cd ./src/recog/    
$  chmod +x run.sh    
$  ./run.sh  dir1  dir2        （dir1：待处理的图片目录， dir2：保存剪影图片的目录）    


##4. 运行例子

 ./run.sh  /home/dailiang/train-origin-pics/   /home/dailiang/profiles-pics/    
在本地CPU-i5-2400, 3.1GHz*4，3.8GiB内存配置下，处理5388张图片的时间约为102min。    

**注意事项**：     
- 该 shell 脚本将开启四个进程，如果程序运行中间中断，请手动将后台的三个 python 进程关闭    
- 产生的结果是与原图尺寸相同的256灰度级的单通道 jpg 图片
