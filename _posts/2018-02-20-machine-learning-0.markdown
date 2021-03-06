---
layout: post
title:  "Machine Learning - 0. Introduction"
date:   2018-02-20 00:25:54 +0900
categories: Machine_Learning

---

안녕하세요, 블로그를 만들고 첫 게시물을 올린지도 벌써 5일이나 지났네요.

그 동안 저는 처음으로 정리할 책을 고르고 있었습니다.

아무래도 어려운 책을 먼저 정리하기 보다는 입문 단계부터 시작하는 것이 좋을 것이라 생각했기 때문에 많은 기계학습 교재 중에 그나마 쉽다고 생각하는 Learning from Data로 결정하였습니다.

혹시나 이 책을 찾는 분이 계실까봐 책의 사진도 아래에 첨부하였습니다.




<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36386209-58125072-15d8-11e8-962c-58a0d3acaf53.jpg"></center>
<br>

이 책은 칼텍의 Abu-Mostafa 교수님이 쓰신 책입니다. 물론 칼텍의 기계학습 수업에서는 이 책을 사용하고 있습니다.

친절하게도 이 책을 사용한 수업의 강의자료는 [칼텍 홈페이지](https://work.caltech.edu/textbook.html)에 모두 무료로 공개되어 있고 (심지어 과제와 답까지!) 강의 영상까지도 [유튜브](https://www.youtube.com/watch?v=mbyG85GZ0PI&list=PLD63A284B7615313A)에 전부 올라와 있기 때문에 독학으로 기계학습을 공부하시는 분들에게 큰 도움이 될 것 같습니다.

물론 칼텍 수업이다보니 영어로 강의하시지만, 친절하게도 영어 자막이 지원되기 때문에 저처럼 영어 듣기에 자신이 없으신 분들도 천천히 따라가시다 보면 큰 문제는 없으실 것입니다.

이 책의 장점을 말하자면 일단 굉장히 짧습니다. (-_-)

보통 기계학습을 배우는 사람들에게 추천되는 책은 대표적으로 Murphy 책이나 Bishop 책 등이 있는데요, 물론 이 책들 모두 좋은 책들입니다만 이 책들의 분량은 정말 상상을 초월할 정도로 길어서 이 책으로 시작을 하신다면 읽다 지쳐 학을 떼실 거라 생각합니다.

Murphy의 Machine Learning : Probabilistic Perspect는 무려 1200여 페이지를 자랑하고, Bishop의 Pattern Recognition and Machine Learning도 700페이지에 달하는 두꺼운 책입니다.

다행히도 이 책들이 워낙 유명하다보니 Murphy 책은 국내 번역본이 존재하고, Bishop 책은 인터넷을 뒤져보면 개인이 직접 번역한 번역 게시물이 있던걸로 기억하는데 그 점을 감안해도 페이지수가 너무 많다보니 입문자들이 읽기에는 부담되는 것이 사실입니다.

하지만 제가 선택한 이 책, Learning from Data는 Chapter 1부터 Epilogue까지 전부 합쳐도 180 페이지밖에 안됩니다! (애초에 책의 부제도 A short course입니다)

이정도의 페이지 수라면 영어 원서라고 해도 크게 부담되는 양은 아니라고 생각합니다.

물론 그렇다고 이 책이 완벽한 것은 아닙니다. 책이 짧은 만큼 Murphy 책이나 Bishop 책과 비교하면 설명이 미흡한 부분이 존재하고, 심지어는 강의자료에는 있지만 책에는 없는 내용까지 존재합니다.

위에서 언급한 칼텍 강의자료/유튜브 강의 영상을 보시면 Neural Networks(Lecture 10), Support Vector Machines(Lecture 14), Kernel Methods(Lecture 15), Radial Basis Functions(Lecture 16) 강의가 분명 존재하지만, 책에는 이 부분들이 생략되어 있습니다.

정말 아쉽게도 이 부분은 강의자료와 강의영상만 보시면서 공부하셔야 합니다. 사실 한 챕터당 약 100분 정도만 강의하시는데 이정도 분량이면 충분히 책에 넣을만 하지 않았을까라는 의문이 들기도 합니다만...

만약 책을 구입하시게 되면 책에 나와있는 비밀 계정(?)으로 Neural Networks와 Support Vector Machines까지는 pdf파일을 제공해주지만, 나머지 두 챕터는 어쩔 수 없이 강의자료와 영상만으로 공부하고 넘어가셔야 합니다.

특히나 국내에서 많이 유명한 책이 아니다보니 구하기도 어렵습니다. 서점들이 직접 이 책을 수입해서 팔질 않아서 아마존이나 알라딘 등을 통해 직구하는 것 밖에 답이 없거든요.

책이 얇아서 하드커버임에도 약 4만원 정도면 구할 수 있긴 하지만, 도착하는 시간이 문제입니다. 저같은 경우는 작년 3월 초에 구입하였으나 3월 말에서야 도착하였습니다 (-_-)

뭐 어쨌든 그나마 짧고 쉬운 책으로 기계학습을 입문하시기에는 이 책만한 것이 없습니다. 구글링을 좀 해보시면 책의 프로그래밍 연습문제도 깃허브에 많이 올라와 있어서 참고 자료도 풍부한 편입니다.

다만 이 책을 통해서 기계학습에 발을 살짝 담구는 것으로 생각하셔야지, 이 책 한 권으로 기계학습을 마스터하겠다! 라는 생각을 하시면 안됩니다. 앞서 말씀드린 것처럼 다른 책들에 비해서 깊게 파고드는 책이 아니거든요.

책의 순서와 칼텍의 강의자료의 순서가 조금 다릅니다만(대표적으로 Linear Model 부분), 강의자료의 내용이 책보다 커버하는 부분이 넓다 보니 챕터별 정리글은 칼텍의 강의자료를 기준으로 할 예정입니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36640391-e4c92b94-1a60-11e8-9d7e-2de0747e35e4.jpg"></center>
<br>

강의의 목차는 위와 같이 총 18개의 챕터로 이루어져 있습니다.

빨간색 부분 챕터는 수학적인 내용 위주, 파란색 부분은 기술적인 내용 위주로 구성되어 있으며 초록색 부분은 개념과 용어에 대해 설명하는 부분입니다.

수학적인 부분 때문에 걱정이 되시는 분들도 계실 텐데요, 학부에서 미적분학, 선형대수학, 공업수학, 통계학 등을 익히셨다면 이해하는데 큰 문제는 없습니다. (다만 첫 강의부터 벡터 연산이 튀어나오니 이것들을 공부 안하셨다면 다소 어려울 수 있습니다)

그럼 최대한 빨리 복습하고 첫 번째 강의자료 정리글로 돌아오겠습니다!