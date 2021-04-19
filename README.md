# 3-tier-Low-Light-Video-Enhancement-Net
<br>

해당 코드는 저조도 개선 분야의 Kindling the Darkness(2019) 라는 논문의 코드를 참고하였습니다 <br>
OpenCV4와 KinD Network를 활용하여 저조도 이미지뿐만 아니라 저조도 동영상을 개선하는 코드를 개발하였습니다 <br>
- Original Code Link : [github](https://github.com/zhangyhuaee/KinD)

<br><br>

* Model Summary <br>
1-tier) video to images(frames) : OpenCV4 <br>
2-tier) LOL images(frames) Enhancement Algorithm : KinD Network <br>
3-tier) images(frames) to video : OpenCV4 <br>
<br><br>

* 요구사항
> Python
>
> Tensorflow >= 1.10.0
>
> numpy, PIL
>
> OpenCV4
<br>

* 바로 테스트 해보기 (저장된 체크포인트 파일로 즉시 test)
> $ python 3-tier_network.py
> 
> test 폴더에 있는 동영상이 result 폴더에 결과로 출력됨을 확인할 수 있습니다
<br>

---- 
<br>
현재 KinD Net의 업그레이드 버전인 KinD++ Net이 나와있습니다
KinD Net의 두번째 서브넷인 Reflectance Restoration Net의 성능을 개선해 전체 네트워크의 성능을 높인 모델입니다 <br>
- Link : [github](https://github.com/zhangyhuaee/KinD_plus)
