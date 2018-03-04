---
layout: post
title:  "Machine Learning - 3. Linear Models I"
date:   2018-03-04 18:26:35 +0900
categories: Machine_Learning
tags: Linear_Classification PLA Pocket_Algorithm Linear_Regression Pseudo-Inverse 
mathjax: true
---

* content
{:toc}

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942324-ba8ff5f4-1fb2-11e8-99d5-cee24a7963e5.png"></center>
<br>




이번 단원에는 새로운 이슈인 선형 모델(Linear Model)에 대해서 알아보는 시간입니다.

Introduction에서도 언급했었지만, 책에서는 Linear Model 파트가 한 챕터로 묶여 있는데요, 온라인 강좌에서는 두 부분으로 나누어 앞부분을 3단원에 넣고, 뒷부분을 9단원에 넣었습니다.

온라인 강의에서도 이 단원을 여기에 넣는 것이 적절하지 않지만, 이론적인 얘기 후에 구체적인 예시를 주고 싶어서 앞에 끼워넣었다고 합니다.

따라서 이 단원은 이전 단원과 다음 단원과는 직접적인 관련은 없습니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942337-f8132cca-1fb2-11e8-9dce-19b28034f744.png"></center>
<br>

이번 단원의 순서는 크게 입력값의 표현을 어떻게 할 것인지부터 시작해서 선형 분류(Linear Classification), 선형 회귀(Linear Regression)를 다루고 비선형 상황에서의 해결책으로 마무리지으며 끝나게 됩니다.

## Input Representation

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942339-fe264b38-1fb2-11e8-8da2-cd2c82ea8bb6.png"></center>
<br>

간단한 예제로 사람이 손으로 쓴 숫자를 분류하는 문제를 들어봅시다.

입력값은 물론 위의 슬라이드에 나온 대로 사람들이 직접 손으로 쓴 숫자들을 따와서 사용하게 됩니다.

그럼 입력값은 그림인데, 어떻게 수식으로 직접 계산할까요?

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942340-034d09b2-1fb3-11e8-871a-3c8ccaefa453.png"></center>
<br>

일반적으로 컴퓨터는 그림을 픽셀의 묶음으로 인식하니 해당 숫자 그림의 각 픽셀값을 입력으로 넣어줄 수 있겠네요.

이 예제의 경우, 한 숫자 그림이 차지하는 픽셀이 256개라고 나와있네요. ($x_0$는 1단원에서 나왔던 Threshold를 위해 만든 Input이었던 것 기억하시죠?)

선형 분류의 대표적인 방법이 바로 1단원에서 배웠던 Perceptron Learning Algorithm (PLA)입니다.

그럼 이 예제를 PLA로 푼다고 가정했을 때, Input Vector의 크기만큼 Weight Vector가 필요하니까 Weight Vector도 256+1 차원 만큼이 필요하겠네요.

물론 이렇게 놓고 문제를 풀 수도 있지만... 사실 이런 간단한 문제에 이렇게 큰 벡터를 사용하기엔 뭔가 이상하네요. 솔직히 데이터를 256차원으로 표현한다고 해봤자 의미없는 입력값이 대부분일 것 같은데 말입니다.

그럼 이 데이터를 어떻게 간단하게 바꿀까요?

가만 생각해보면 숫자의 경우 0, 1, 8 같은 숫자는 대칭적이고 그 외의 숫자는 비대칭이니 대칭 여부를 통해 숫자를 분류할 수 있을 거 같네요.

또 1, 7에 비해서 8은 숫자 그림에서 차지하는 검은색 픽셀이 더 많겠군요? 이것도 숫자를 분류하는 데 쓸 수 있겠네요.

그럼 간단하게 이 두 개의 특징만을 사용해서 Input Vector를 2+1 차원으로 줄여봅시다!

## Linear Classification

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942342-09442cba-1fb3-11e8-8614-72e46e3411c5.png"></center>
<br>

이전 슬라이드에서 정했던 간단한 Input Vector로 숫자 1과 5를 분류한 그림입니다.

아무래도 1보다 5가 차지하는 검은색 픽셀이 더 많고, 숫자 1은 대칭적인데 비해 5는 대칭적이지 않으니 이 둘을 구분하는건 크게 어렵지 않아 보입니다.

위 슬라이드를 보시면 대략적으로는 구분이 가능하지만, 몇몇 지점에서 약간의 Noise가 있음을 알 수 있습니다.

이 Noise 때문에 이 데이터는 선형 분리(Linearly Seperable)이 불가능하네요. 어이쿠! PLA는 선형 분리가 불가능하면 수렴하지 않는다고 했는데, 그럼 이 문제는 어떻게 풀죠?

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942343-0e275874-1fb3-11e8-9891-7548a138aede.png"></center>
<br>

실제로 이전 슬라이드의 데이터를 PLA를 사용한 결과입니다.

당연히 데이터가 선형 분리가 불가능하니 수렴하지 않고 무한히 반복되겠죠. 그래서 어쩔 수 없이 딱 1000번만 수행하도록 제약을 걸어둔 모습입니다.

PLA를 수행하는 도중 데이터를 구분하는 선(다시말해 Weight Vector의 값)이 계속 움직이기 때문에 $E_{in}$ (In sample Error)와 $E_{out}$ (Out of sample Error)가 커졌다가 작아졌다 하는 것을 알 수 있네요.

물론 실제로는 $E_{out}$를 알지 못합니다. 이 예제에서는 $E_{in}$와 $E_{out}$가 어떤 관계가 있는지 보여주기 위해서 특별히 같이 그려준 겁니다.

다행히도 $E_{in}$가 커지면 $E_{out}$도 커지고, 작아질 때도 같이 작아지는 비례 관계임을 알 수 있네요. 즉, 앞으로는 $E_{in}$만 알아도 이를 낮추는 방향으로 설계를 한다면 $E_{out}$도 자연스레 줄어드는 결과가 나온다는 좋은 정보를 알게 되었습니다.

왼쪽 그래프를 보니 수행 횟수가 1000번일 때의 Error가 250번 정도일 때보다 크게 나타나네요. 하지만 이 알고리즘은 1000번 수행 후의 결과를 출력하다 보니 이전에 더 좋은 성능을 보이는 Weight Vector를 찾았음에도 그보다 못한 값이 최종 값으로 확정되어버렸네요.

이걸 개선할 방법이 없을까요?

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942344-16fbf892-1fb3-11e8-98bc-10285e42de00.png"></center>
<br>

여기서는 이를 <b>Pocket Algorithm</b>으로 해결합니다.

Pocket Algorithm은 수행 중에 최고 성능을 보이는 Weight Vector 값을 저장해 두고, Weight Vector가 변경되었을 시 성능이 기존보다 떨어진다면 이를 반영하지 않는 알고리즘입니다.

즉, Pocket Algorithm을 사용한다면 Weight Vector를 학습시킬수록 Error가 높은 (낮은 성능을 보이는) Vector가 나온다고 해도 항상 최선의 결과만 출력해 줄 수 있다는 장점이 있겠네요.

이 예제에서는 수행 횟수가 250번 근처일 때 가장 성능이 좋고, 이후에는 계속 이보다 나쁜 성능을 가지는 Vector만 나오다 보니 Pocket Algorithm의 경우 250번 이후부터 변화가 없는 것을 알 수 있습니다. (오른쪽 그림)

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942345-1e0a54e4-1fb3-11e8-8c6e-d1e539224ad5.png"></center>
<br>

위 슬라이드는 PLA를 사용한 결과와 Pocket Algorithm을 사용한 결과를 비교한 그림입니다.

확실히 Pocket Algorithm으로 나눈 결과가 더 바람직해 보인다는 것을 알 수 있습니다.

## Linear Regression

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942347-23f2be46-1fb3-11e8-88c2-86eb63e79f61.png"></center>
<br>

지금까지 선형 분류를 하는 방법을 알아보았습니다.

이제 선형 회귀를 하는 방법을 알아보겠습니다. "회귀"라는 것은 분류와는 다르게 출력 결과가 $+1/-1$이 아니라 실수값으로 나오는 것을 말합니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942351-29232072-1fb3-11e8-9daf-ce0e240c8690.png"></center>
<br>

이전 단원에서 다루었던 카드를 발급하는 문제로 돌아와봅시다.

그때는 신청자의 정보를 바탕으로 카드를 발급해 줄 것인가/거부할 것인가의 여부를 다뤘다면, 이제는 카드를 발급해준다면 그 카드의 한도를 얼마로 정해줄 것인가를 다룬다고 보시면 됩니다.

물론, 지원자의 정보의 형태(Input Vector)는 분류를 할 때와 크게 다르지 않습니다.

단, 분류를 할 때는 Hypothesis $h$의 형태가 $sign(\mathbf{w}^{\sf T} \mathbf{x})$ 였지만, 이제는 실수값의 출력이 나와야 하므로 $sign()$ 함수를 떼버린 $\mathbf{w}^{\sf T} \mathbf{x}$ 만 남게 됩니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942353-2df3752a-1fb3-11e8-8ee9-4fefdf7247b2.png"></center>
<br>

이전 슬라이드에서 언급했다시피 입력 값은 분류때와 큰 차이점은 없습니다.

다만 분류에서는 $y_n$의 값이 $+1/-1$이었지만 회귀에서는 실수 값이 들어간다는 것만 주의하시면 됩니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942354-336f5c76-1fb3-11e8-88df-76deeac60b92.png"></center>
<br>

문제가 다르니 Error를 어떻게 측정할 것인가 도 생각해 보아야겠네요.

분류에서는 출력이 $+1/-1$ 뿐이다 보니 그냥 정답과 다른 것의 갯수만 세면 충분했습니다. (그래서 전체 중에 몇개나 틀렸는지를 비율로 표시했죠)

하지만 이 경우에는 출력의 값이 무한정 많다보니 단순히 틀렸냐 틀리지 않았냐만을 따지기에는 곤란합니다.

예를 들어 정답 출력이 1000 인 입력값에 대해서, 950 정도 예측했다면 살짝 틀린 것이지만 5000으로 예측했다면 그건 크게 실수한 것이니까요.

따라서 각 입력값에 대해 얼마나 "큰 차이"로 틀렸냐를 토대로 Error를 측정하게 됩니다.

여기서는 Hypothesis Function과 Target Function의 차이를 제곱한 값을 사용하게 됩니다. 이를 <b>Squared Error</b>라고 합니다.

이걸 보시면 의문이 드실 수도 있습니다. 단순한 차이를 원한다면 제곱할 필요 없이 그냥 차이의 절대값을 쓰면 되는게 아닌가... 라고 말입니다.

제곱을 쓰는 데는 여러가지 이유가 있으나 대표적인 이유로는 절대값을 쓰게 되면 미분이 힘들어지기 때문입니다. (차이가 크면 클수록 패널티를 많이 주기 위해서라고 하셔도 맞습니다)

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942356-3973d944-1fb3-11e8-863b-6f9c28fa69e3.png"></center>
<br>

이 Error Measure를 그림으로 표현한 예시입니다.

왼쪽 그림은 Input Vector가 1차원일 때의 예시이며, 오른쪽 그림은 Input Vector가 2차원일 때의 예시입니다.

파란색 선(오른쪽 그림에서는 파란색 평면)이 의미하는 것은 Hypothesis Function $h$이고, 빨간색 선이 실제 분류값(Target Fucntion $f$의 값)과 예측한 결과 값의 차이입니다.

가끔 헷갈리시는 분들이 있는데, 빨간색 선을 그릴 때 $h$에 대해서 직교하는 선을 그리는 것이 아님을 유의하셔야 합니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942357-3e62cd7a-1fb3-11e8-8993-b39c9d161efa.png"></center>
<br>

그럼 이 Error Measure를 통해서 $E_{in}$을 계산해봅시다.

입력값의 개수가 $N$이므로 모든 입력값에 대해 평균적인 차이를 계산한다면 아래와 같습니다.

$$
E_{in}(\mathbf{w}) = \frac{1}{N} \sum_{n=1}^N (\mathbf{w}^{\sf T} \mathbf{x}_n - y_n)^2
$$

식을 좀 더 간단하게 표현하기 위하여, $\mathbf{x}_n$들을 묶어 $\mathbf{X}$라는 큰 벡터로 표현해봅시다. 그리고 출력값인 $y_n$도 하나로 묶어 $\mathbf{y}$로 묶어보겠습니다.

이렇게 바꾸면 Sigma 연산기호가 사라지고, $\mathbf{X}$ 벡터와 $\mathbf{w}$ 벡터의 곱 연산으로 간단하게 나타낼 수 있습니다.

따라서 아래와 같이 간단한 꼴로 표현이 되겠네요.

$$
E_{in}(\mathbf{w}) = \frac{1}{N} \lVert \mathbf{X}\mathbf{w} - \mathbf{y} \rVert^2
$$

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942360-4443e3be-1fb3-11e8-80c4-41a1304a40dc.png"></center>
<br>

우리가 원하는 것은 $E_{in}$의 최소값입니다.

어떤 함수의 어디에서 최소값을 갖는지 구하는 방법은 그 함수의 도함수(Derivative)를 구해서 그 값이 0인 점을 찾으면 되는 것을 다들 아실겁니다. (물론 모든 함수의 최소값이 도함수가 0인 점이라는 것은 아닙니다만, $E_{in}$ 함수에서는 해당 지점이 최소값입니다)

$E_{in}$을 미분하려면 Vector Calculus를 알아야 하는데, 모르시는 분을 위해서 중간 과정을 조금 적어드리겠습니다.

$$
E_{in}(\mathbf{w}) = \frac{1}{N} \lVert \mathbf{X}\mathbf{w} - \mathbf{y} \rVert^2
$$

위의 원래의 식에서 제곱을 없애기 위해 $\lVert \rVert$ 부분을 풀어써 보겠습니다.

$$
E_{in}(\mathbf{w}) = \frac{1}{N} (\mathbf{X}\mathbf{w} - \mathbf{y})^{\sf T} (\mathbf{X}\mathbf{w} - \mathbf{y})
$$

엇... Transpose가 어디서 튀어나왔나 싶으실텐데 이것은 Transpose를 붙이지 않으면 행렬 곱을 할 수 없기 때문에 붙여진 것이라고 생각하시면 됩니다.

()위에 붙어있는 Trasnpose가 보기 싫으니, Transpose를 () 안으로 넣어봅시다.

$$
E_{in}(\mathbf{w}) = \frac{1}{N} (\mathbf{w}^{\sf T} \mathbf{X}^{\sf T} - \mathbf{y}^{\sf T}) (\mathbf{X}\mathbf{w} - \mathbf{y})
$$

이 식을 전개한다면,

$$
E_{in}(\mathbf{w}) = \frac{1}{N} (\mathbf{w}^{\sf T} \mathbf{X}^{\sf T} \mathbf{X} \mathbf{w} - 2 \mathbf{y}^{\sf T} \mathbf{X} \mathbf{w})
$$

이렇게 쓸 수 있습니다. $E_{in}$은 $\mathbf{w}$에 대한 함수이므로 $\mathbf{w}$로 미분을 하게 되면($\nabla E_{in}(\mathbf{w})$) 아래와 같이 도함수가 0이 되는 $\mathbf{w}$를 찾는 문제로 바뀌게 됩니다.

$$
\frac{2}{N} \mathbf{X}^{\sf T} (\mathbf{X}\mathbf{w} - \mathbf{y}) = 0
$$

여기서 $ \frac{2}{N} $은 의미없는 상수이므로 지워주게 되면,

$$
\mathbf{X}^{\sf T} (\mathbf{X}\mathbf{w} - \mathbf{y}) = 0
$$

괄호 ()를 없애기 위해 분배법칙을 사용해 전개해줍니다.

$$
\mathbf{X}^{\sf T}\mathbf{X}\mathbf{w} - \mathbf{X}^{\sf T}\mathbf{y} = 0
$$

$ \mathbf{X}^{\sf T}\mathbf{y} $항을 오른쪽으로 넘겨주면,

$$
\mathbf{X}^{\sf T}\mathbf{X}\mathbf{w} = \mathbf{X}^{\sf T}\mathbf{y}
$$

왼쪽 항에 $\mathbf{w}$만 남기기 위해서는 $\mathbf{X}^{\sf T}\mathbf{X}$의 역행렬을 양쪽에 곱해주어야 합니다. 일단 $\mathbf{X}$ 자체는 정사각행렬(Square Matrix)이 아니지만, 임의의 행렬에 대해서 그 행렬의 Transpose 행렬을 곱해주면 정사각행렬이 되므로 $\mathbf{X}^{\sf T}\mathbf{X}$는 정사각행렬입니다.
(임의의 행렬이 크기가 $n \times m$ 이라 했을 때, 이 행렬을 Transpose 해주면 $m \times n$이 되고, ($m \times n$ 행렬)($n \times m$행렬) 연산을 해주면 $m \times m$ 행렬이 나오므로 $m$과 $n$에 관계없이 무조건 정사각행렬이 됩니다)

이제 $\mathbf{X}^{\sf T}\mathbf{X}$ 행렬이 역행렬이 존재하는지 확인해야 하는데, 만약에 이 행렬의 역행렬이 존재하지 않는다면 위의 식 자체가 의미가 없어져 버리므로, 여기서는 있다고 가정하고 계산하겠습니다.

양쪽 항에 $\mathbf{X}^{\sf T}\mathbf{X}$의 역행렬을 곱해주면,

$$
\mathbf{w} = (\mathbf{X}^{\sf T}\mathbf{X})^{-1}\mathbf{X}^{\sf T}\mathbf{y}
$$

이 성립하게 됩니다. 여기서 $\mathbf{X}$와 관련된 항인 $(\mathbf{X}^{\sf T}\mathbf{X})^{-1}\mathbf{X}^{\sf T}$를 묶어서 $\mathbf{X}^{\dagger}$ 이라 한다면 아래처럼 깔끔하게 바뀌겠지요.

$$
\mathbf{w} = \mathbf{X}^{\dagger}\mathbf{y}
$$

여기서 $\mathbf{X}^{\dagger}$를 $\mathbf{X}$의 <b>Pseudo-Inverse</b> (의사역행렬)라고 합니다. 왜 이런 이름이 붙였냐면, 이 행렬은 $\mathbf{X}$의 역행렬은 아니지만, 역행렬처럼 $\mathbf{X}^{\dagger}\mathbf{X} = I$가 나오는 성질을 가지거든요.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942361-4c5ba654-1fb3-11e8-861e-7f4c0415f15c.png"></center>
<br>

이전 슬라이드를 보시고 이게 뭥미... 하고 당황하신 분들도 걱정하지 마세요. Pseudo-Inverse의 전개 과정을 안다면 더욱 좋겠지만 모르셔도 전체적인 방법을 이해하는데는 큰 무리가 없습니다.

아셔야 할 것은 "회귀 문제를 풀기 위해서는 Pseudo-Inverse를 계산해야 하는구나" 정도만 아시면 됩니다. 어차피 이걸 계산할 줄 몰라도 요즘엔 MATLAB이나 Mathematica가 다 계산해주거든요(...)

어쨌든 Pseudo-Inverse가 어떤 크기를 갖는지 알아몹시다.

$\mathbf{X}$는 $d$차원의 입력이 $N$개 만큼 있으니 $N \times (d+1)$ 크기의 행렬이 될 것입니다. (왜 $d$가 아니라 $d+1$이냐면 Threshold를 나타내는 $x_0$가 포함되기 때문입니다)

그럼 $\mathbf{X}^{\sf T}$는 이 반대로 $(d+1) \times N$ 크기의 행렬이니 $\mathbf{X}^{\sf T} \mathbf{X}$ 를 계산한다면 $(d+1) \times (d+1)$ 크기의 행렬이 되겠네요.

역행렬을 구한다고 해도 행렬의 크기는 변하지 않으니 $(\mathbf{X}^{\sf T} \mathbf{X})^{-1}$ 또한 $(d+1) \times (d+1)$ 크기의 행렬이겠네요.

$\mathbf{X}^{\sf T}$행렬은 $(d+1) \times N$ 크기라고 했으니 $(\mathbf{X}^{\sf T} \mathbf{X})^{-1} \mathbf{X}^{\sf T}$를 계산하면 최종적으로 Pseudo-Inverse $\mathbf{X}^{\dagger}$는 $(d+1) \times N$ 크기의 행렬이 된다는 것을 알 수 있습니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942364-51ed6fbc-1fb3-11e8-9498-53d0429296b0.png"></center>
<br>

최종적으로 선형 회귀를 하는 알고리즘을 정리해보면, 첫째로 입력값과 그 정답을 나타내는 값을 각각 $\mathbf{X}$와 $\mathbf{y}$ 행렬로 묶어주고, 둘째로 Pseudo-Inverse $\mathbf{X}^{\dagger}$를 계산한다음, 마지막으로 Pseudo-Inverse $\mathbf{X}^{\dagger}$와 최종적으로 선형 회귀를 하는 알고리즘을 정리해보면, 첫째로 입력값과 그 정답을 나타내는 값을 각각 $\mathbf{X}$와 $\mathbf{y}$ 행렬로 묶어주고, 둘째로 Pseudo-Inverse $\mathbf{X}^{\dagger}$를 계산한다음, 마지막으로 Pseudo-Inverse $\mathbf{X}^{\dagger}$와 $\mathbf{y}$를 곱해주면 $\mathbf{w}$를 알 수 있게 됩니다.

이걸 보고 "아니 이게 왜 Learning이야? 한번만 계산하면 끝나는걸!" 이라고 생각하실 수도 있습니다. (사실 저도 그랬습니다)

다만 책의 저자이신 Abu-Mostafa 교수님께서는 꼭 PLA처럼 모든 데이터에 대해 하나하나 Weight Vector를 수정하는 것만이 Learning이 아니라고 합니다.

다시 말해서 어떻게 구하는지 그 과정은 별로 중요한 것이 아니라고 합니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942367-577654b2-1fb3-11e8-8428-1d61eb2ca9ee.png"></center>
<br>

선형 회귀는 결과 값이 실수로 나온다는 것을 알고 있습니다.

그런데 생각해보니 선형 분류에서 나오는 결과는 $+1/-1$인데, 이것도 실수니까 회귀와 관련 있지 않을까? 라는 생각이 들겁니다.

다시 말해, 선형 회귀에서 사용한 Weight Vector $\mathbf{w}$를 가지고 분류를 할 수 있지 않을까 라는 겁니다.

안타깝게도 회귀와 분류는 그 목적이 다르기 때문에 완벽하게 일치하지는 않습니다.

다만, 기존에 분류를 할 때 $\mathbf{w}$의 초기값을 무작위 값으로 정의하였지만, 회귀에서 사용했던 $\mathbf{w}$를 초기값으로 정하게 된다면 분류에 수렴하기 까지 시간이 매우 단축되는 결과를 얻을 수 있습니다. 

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942369-5c0d2820-1fb3-11e8-8596-b7de16574ac1.png"></center>
<br>

선형 분류 문제에서 선형 회귀의 Weight Vector $\mathbf{w}$를 초기값으로 설정해 그림으로 표현한 결과입니다.

완벽하지는 않지만, 정답과 매우 유사한 분류가 이루어졌음을 알 수 있네요.

이제 저기서 PLA를 사용하게 되면, 머지않아 곧 최적의 $\mathbf{w}$이 구해지겠구나라는 것을 예상할 수 있습니다. 

## Nonlinear Transformation

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942370-612d9de4-1fb3-11e8-925b-82c632d76f59.png"></center>
<br>

이제 마지막으로 비선형(Non-Linear) 문제는 어떻게 해결하는지를 알아볼 예정입니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942371-6717d134-1fb3-11e8-9abd-99b6365b094b.png"></center>
<br>

선형 분류/회귀는 굉장히 편리하고 효과가 좋은 방법이지만, 안타깝게도 모든 문제가 선형으로 해결되지는 않습니다.

간단한 예제로 위 슬라이드의 왼쪽 그림과 같은 데이터의 경우, 어떻게 나누어도 선형으로 나눌 수가 없습니다.

오른쪽 그림처럼 원 모양으로 표현해야 제대로 나누어질텐데, 선형 방법으로는 저렇게 원 형태를 표현할 수 없으니 참으로 난감합니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942373-6bc1c096-1fb3-11e8-9f51-e63fa11163ad.png"></center>
<br>

카드 발급 문제로 돌아가서 "해당 집에서 거주한지 얼마나 오래되었는지(years in residence)" 라는 요소를 생각해봅시다.

왜 그런지 이유는 모르겠지만, 한 집에서 너무 적게 거주했거나(1년 미만) 너무 오래 거주한 경우(5년 초과) 부정적으로 평가한다고 합니다.

이런 경우 선형 모델을 통해 표현할 수 있을까요?

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942374-7076d3a6-1fb3-11e8-8b3a-0e2f1d25b866.png"></center>
<br>

이 문제에 답을 하기 위해서는 "무엇"에 선형이냐 라는 것부터 생각해보아야 합니다.

간혹 입력 값이 선형이기 때문에 선형 모델이라고 생각하시는 분들이 있는데, 선형 모델이라고 이름이 붙은 이유는 "<b>Weight</b>"에 대해 선형이기 때문에 선형 모델이라고 부르는 겁니다.

그러니까, 다시 말해서 입력값 $\mathbf{x}$은 선형이든 아니든 크게 상관이 없다는 겁니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/36942376-76712324-1fb3-11e8-991f-ba3e3e4973ff.png"></center>
<br>

자, 그럼 입력값이 선형이든 아니든 상관이 없다고 했으니, 입력 값을 우리 맘대로 한번 바꾸어봅시다.

임의의 함수 $\Phi$를 정의해서 입력값 $\mathbf{x}$ 제곱하는 연산을 수행하게 만들어 봅시다.

위 슬라이드의 왼쪽 그림은 20번 슬라이드에 나왔던 그 예제입니다. 그런데 모든 데이터에 대해 $\phi$ 연산을 적용시키니 오른쪽 그림처럼 아주 이쁘게 변했네요. 이 바뀐 데이터의 분포는 선형 분리까지 가능하네요!

아하, 그럼 선형 분리가 도저히 안되는 데이터의 경우 데이터를 비선형으로 바꾸면 되겠군요?

물론 맞는 말이긴 합니다만 여기에는 한 가지 문제가 있습니다. 위의 그림처럼 간단한 예제의 경우에는 적당한 $\Phi$를 쉽게 구할 수 있지만, 일반적인 상황에서 적절한 $\Phi$를 찾는 것이 쉬울까요?

그렇지 않다는 것을 단번에 아실 수 있겠죠? 이 문제에 대한 해결 방법은 추후에 다시 언급됩니다.

이번 단원은 여기까지입니다. 감사합니다.


<br><br>

|<b><center>저도 배우고 있는 학생이라 잘못 이해한 부분이 있을 수 있습니다.</center></b>|
|<b><center>질문이나 틀린 부분에 대한 지적을 댓글로 달아주시면 최대한 빨리 답변해드리겠습니다.</center></b>|