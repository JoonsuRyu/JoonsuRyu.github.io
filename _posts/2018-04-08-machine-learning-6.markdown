---
layout: post
title:  "Machine Learning - 6. Theory of Generalization"
date:   2018-04-08 14:00:57 +0900
categories: Machine_Learning
tags: Hoeffding's_Inequality Growth_Function Break_Point
mathjax: true
---

* content
{:toc}

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828621-5fbc477c-2edf-11e8-9dd3-d3b94cf42c51.png"></center>
<br>





안녕하세요, 정말 오랜만에 글을 쓰게 됬네요.

사실 학기중에는 주말밖에 글을 쓸 시간이 없는데 주말에 일이 생기면 그 주는 그냥 날리게 되버려서...

거의 한달만에 올리게 됬네요. 앞으로는 최대한 매주 한개는 쓸 수 있도록 노력해 보겠습니다.

이번 단원은 지난 단원에서 다루었던 $m_{\mathcal{H}}$에 대한 증명들을 다루게 됩니다.

증명에서 눈치채셨을 수도 있으시겠지만, 이 단원은 가장 이론적인 단원입니다. 즉, 수학식이 굉장히 많이 나옵니다...

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828650-7bdb0754-2edf-11e8-8e83-745c5a3c687e.png"></center>
<br>

이번 단원의 증명은 크게 두 가지를 다루게 됩니다.

첫째는 $m_{\mathcal{H}}$가 다항함수인 것을 증명하는 것이고, 두번째는 $m_{\mathcal{H}}$가 Hoeffding's Inequality에서 $M$을 대체할 수 있는가에 대한 증명입니다. 

## Proof that $m_{\mathcal{H}}(N)$ is polynomial

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828664-81c999fa-2edf-11e8-89b0-727c3557710f.png"></center>
<br>

$m_{\mathcal{H}}(N)$이 $N$에 대해서 다항함수임을 증명하려면, 부등식을 통해 $m_{\mathcal{H}}(N)$이 $N$으로 이루어진 다항식보다 작거나 같다는 것을 증명하면 됩니다.

다항식의 차수가 크면 어떡하지? 라는 생각이 드실 수도 있는데, 다항식의 차수가 크든 작든 $N$이 충분히 커지게 되면 어차피 지수함수꼴보다 항상 작아지기 때문에 큰 문제는 없습니다.

이 증명은 조금 복잡하기 때문에, 새로운 함수를 하나 정의합니다. 

$B(N, k)$라는 함수를 하나 정의하는데, 이것은 $N$개의 점이 있고 Break Point가 $k$일 때 가능한 Dichotomie의 최대 개수를 의미합니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828667-86f35f38-2edf-11e8-88de-43d31445612e.png"></center>
<br>

이 $B(N, k)$ 함수를 구체적인 변수가 포함된 값으로 정의하기 위해 한번 위 슬라이드의 오른쪽 표를 보면서 따져보겠습니다.

오른쪽 표에서 $$\mathbf{x}_{1}, \mathbf{x}_{2}, ... \mathbf{x}_{N}$$은 각각 $N$개의 점을 의미합니다. 이 값은 $+1$이나 $-1$의 값을 가질 수 있습니다.

가능한 Dichotomy 들의 조합 중에서, $$\mathbf{x}_{1} ~ \mathbf{x}_{N-1}$$ 까지의 값이 같고, $$\mathbf{x}_{N}$$의 값만 정 반대인 집합을 $$S_{2}$$라고 정의하고, 그렇지 않은 것들을 모아둔 집합을 $S_{1}$이라고 정의하겠습니다.

그럼 모든 Dichotomy 들의 조합들은 $S_{1}$이나 $S_{2}$ 둘 중 하나에 포함될 것이라는 것을 쉽게 알 수 있습니다.

그 중에서 $S_{2}$를 좀더 세분화하기 위해, $$\mathbf{x}_{N}$$의 값이 $+1$인 것들의 집합을 $$S^{+}_{2}$$이라 하고, $$\mathbf{x}_{N}$$의 값이 $-1$인 것들의 집합을 $$S^{-}_{2}$$라고 하겠습니다.

당연히 $$S^{+}_{2}$$와 $$S^{-}_{2}$$의 갯수는 같을 수밖에 없겠죠. 이 집합의 원소의 갯수를 $$\beta$$라고 하고, 아까 언급했던 집합 $$S_{1}$$의 갯수를 $\alpha$라고 하겠습니다.

따라서 $B(N, k)$의 값은 $\alpha + 2\beta$로 표현이 가능합니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828672-8c06fbec-2edf-11e8-98b5-0d3c0c88d5c4.png"></center>
<br>

이제 점 $$\mathbf{x}_{N}$$을 삭제해보겠습니다. 다시말해 $N-1$개의 점에서 Break Point가 $k$일 때 $B(N-1, k)$를 구하자는 것입니다.

그런데 "어? 그럼 그냥 $\alpha + \beta$가 되는 것일텐데, 왜 등호가 아니라 $B(N-1, k)$인거지?" 라고 생각하실 수도 있을텐데요,

$B(N-1, k)$는 가능한 Dichotomy의 <b>최대</b> 개수라고 정의되었습니다. 그러나 방금처럼 점이 $N$개였던 표에서 $$\mathbf{x}_{N}$$만 지운 표가 점이 $N-1$일 때의 최대 개수인지 아닌지 모르기 때문에 부등호가 들어간 것입니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828676-9094f2cc-2edf-11e8-836f-ec9de447034d.png"></center>
<br>

마찬가지로 이번에는 $B(N-1, k-1)$을 구해볼 것인데, 역시 점이 $N-1$개이므로 표에서 점 $$\mathbf{x}_{N}$$을 삭제해보겠습니다.

여기서는 이전 슬라이드와 마찬가지로 $B(N-1, k-1)$가 $\beta$ 이상이라고 부등호가 되어있는데, 왜 이것이 $k-1$의 Break Point를 가지는지 궁금해하시는 분들이 많을 것 같습니다.

만약에 $\beta$가 $k$의 Break Point를 가진다고 가정한다면, Break Point의 정의에 의해서 $2^{k-1}$개의 Dichotomy를 표현 가능해야 합니다.



<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828682-9771d0c4-2edf-11e8-8d6a-de4d48bc9f02.png"></center>
<br>

이전의 두 슬라이드의 내용을 정리해보면 다음과 같습니다.

$$
B(N, k) = \alpha + 2\beta
$$

$$
\alpha + \beta \leq B(N-1, k)
$$

$$
\beta \leq B(N-1, k-1)
$$

따라서 위 식에서 두번째 식과 세번째 식을 양변을 더해주면 아래와 같은 식을 유도할 수 있습니다.

$$
B(N, k) \leq B(N-1, k) + B(N-1, k-1)
$$

이를 토대로 임의의 $N$, $k$에 대해서 $B(N, k)$의 값을 Recursive하게 계산할 수 있다는 것을 알 수 있습니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828685-9c3a75f2-2edf-11e8-8e51-15f25031672a.png"></center>
<br>

방금 유도한 식을 토대로 $N$과 $k$에 따라 $B(N, k)$값이 어떻게 변하는지 위 슬라이드의 오른쪽 표에 나와있습니다.

표를 채우려면 먼저 $B(N, 1)$과 $B(1, k)$를 구해야 하는데, $B(N, 1)$은 점이 몇개가 주어지든 Break Point가 1이란 얘기니까 무조건 딱 한가지로만 분류가 가능하다는 것이겠죠. 따라서 $B(N, 1)=1$이 됩니다.

이번엔 $B(1, k)$를 계산해야 하는데, 점이 딱 1개만 있으면 어차피 나눌 수 있는 경우의 수는 $+1$ 또는 $-1$ 밖에 없으니까 Break Point가 아무리 커봤자 $B(1, k)=2$라는 것을 쉽게 알 수 있죠. (단, $k=1$일 때 제외)

이렇게 첫번째 행/열만 채우게 되면 나머지는 이전 슬라이드에서 보였던 부등식을 통해 채울 수 있습니다.

이 표의 값들을 보실 때 주의하실 점은, $B(N, k)$의 정확한 값이 아니라 <b>상한(Upper Bound)</b>이라는 겁니다. 애초에 유도한 식 자체가 부등식이기 때문이죠.

즉 표의 값을 읽을 때, 예를 들면 표에서 붉은 글씨를 확인해 보시면 $B(3, 2)$이 $4$라고 나와있습니다. 이 말은 $B(3, 2)$가 정확하게 $4$라는 뜼이 아니라 아무리 커봤자 최대가 $4$라는 뜻입니다. 물론 $4$보다 작을 수도 있겠죠. (이건 직접 하나하나 계산해보기 전까지는 모릅니다.)

그럼 실제로 $B(3, 2)$의 값이 뭔지 궁금하다구요? 이건 저희가 이미 계산해본 적 있습니다. 이전 단원의 맨 마지막 슬라이드에서 간단한 Puzzle을 풀었었는데, 이 때가 $N=3, k=2$의 예제였습니다. 계산했을 때 값이 $4$가 나왔었는데, 우연히도 $B(3, 2)$의 상한과 같은 값이 나왔네요. 하지만 이것은 우연일 뿐, 항상 이렇게 같은 값이 나오지는 않음에 유의해 주시기 바랍니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828694-a22efa14-2edf-11e8-802a-4e5153461f39.png"></center>
<br>

하지만 이렇게 Recursive하게 계산하는 것은 계산 속도도 오래걸리고 귀찮기 때문에, 이를 한번에 표현할 수 있는 일반항을 찾아야합니다.

위 슬라이드에 나온 것처럼, $B(n, k)$에 대한 일반항을 조합(Combination)들의 합으로 제시하고 이를 증명합니다.

증명 방법은 흔히 사용하는 수학적 귀납법(Induction)으로 증명합니다. 먼저 맨 처음 항이 True임을 보여야 하는데, 이건 그냥 $N=1$일 때와 $k=1$일 때 각각 넣어서 증명하면 간단하니 패스하겠습니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828698-a795686c-2edf-11e8-8362-79358420feb0.png"></center>
<br>

그 다음의 과정이 조금 흥미로운데, 여기서는 $B(N-1, k)$의 합과 $B(N-1, k-1)$의 합이 $B(n, k)$가 되는지를 보였습니다.

첫번째 줄에서 두번째 줄로 넘어갈때는 두 항을 합치기 위해 시그마를 똑같이 $i=1$부터 $k-1$까지 맞춰눈 것이고, 세번째 줄에서 네번째 줄로 넘어간 것은 조합에서 사용하는 파스칼의 삼각형을 사용해서 두개의 조합을 하나로 합친 것입니다.

이 외에는 단순한 계산이므로 설명을 생략하겠습니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828707-acae5200-2edf-11e8-9b71-1a2288225b47.png"></center>
<br>

사실 중요한 것은 이 슬라이드입니다.

이전 슬라이드까지의 과정을 통해서 결국 $m_{\mathcal{H}}$가 조합들로 이루어진 합보다 작다는 것이 증명되었고, 이 조합은 아무리 커봤자 $N^{k-1}$의 항을 가진 다항함수이므로 결과적으로 그토록 원하던 <b>"Growth Function이 다항함수이다"</b> 라는 결론이 나온 것입니다.

지금까지 Hoeffding's Inequality가 무한대에 가까운 $M$으로 고통받았던 것을 생각해보면 Growth Function을 통해 다항함수꼴로 줄인 것은 매우 큰 의미가 있습니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828714-b4130d1a-2edf-11e8-90a6-15c124d907a9.png"></center>
<br>

왜 그것이 큰 의미가 있냐라는 것은 이 슬라이드를 통해 설명할 수 있습니다.

이전 단원에서 Growth Function을 설명할 때 사용한 3가지 예제가 기억나실 겁니다. (Positive Ray, Positive Interval, Convex Set)

이중에서 Positive Ray와 Positive Interval은 지난 단원에서 직접 계산했기 때문에, 방금 유도한 Growth Function의 상한과 비교해 보겠습니다. 운이 좋게 Positive Ray와 Positive Interval은 상한과 똑같이 나왔습니다만, 아까도 말씀드렸듯이 항상 똑같은게 아니라는걸 꼭 기억하셔야 합니다.

그런데 2D Perceptron은 이전에 Break Point가 4인것은 보였지만, Growth Function을 직접 구할 수는 없었습니다. 하지만 방금 유도한 Growth Function 상한을 이용하면 2D Perceptron도 공식에서 나온 다항함수보다 작다는 것을 보일 수 있습니다.

결국 이런 방식을 통해 그 어떠한 케이스에서도 Break Point만 찾으면 Growth Function이 다항함수 꼴로 상한이 정해진 다는 것을 알 수 있습니다.

## Proof that $m_{\mathcal{H}}(N)$ can replace $M$

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828721-ba673d9e-2edf-11e8-93a8-6666580691d0.png"></center>
<br>

그럼 지금까지 $m_{\mathcal{H}}$가 다항함수로 이루어진 식의 상한으로 이루어진 것이 증명되었으니, 이제 정말 중요한 Hoeffding's Inequality에 있던 $M$ 대신에 $m_{\mathcal{H}}$를 대입하기 위한 증명이 필요합니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828727-c216ed78-2edf-11e8-800b-cc89829125a9.png"></center>
<br>

사실 우리가 원하는 결과는 Hoeffding's Inequality에서 $M$ 자리에 그대로 $m_{\mathcal{H}}$가 들어가는건데, 솔직히 생각해보면 $M = m_{\mathcal{H}}$도 아닌데 그렇게 직접적으로 교체될 수는 없겠죠.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828736-c90641b0-2edf-11e8-8ea7-91112d84b9a2.png"></center>
<br>

어쨌든 이 $M$ 대신에 $m_{\mathcal{H}}$를 넣어 (변형된) 식을 증명해야겠지만, 강의에서는 이 증명이 너무 복잡하기 때문에 부록에 따로 빼 놓았다고 합니다.

제가 확인해보니 6페이지 정도 분량에 수식이 꽉 차있던데 솔직히 저도 안읽었습니다... 어차피 너무 길고 굳이 그것까지 읽지 않아도 될거 같았거든요.

이 단원에서는 그냥 간단하게 이렇게 하겠다~ 정도만 언급하고 넘어갑니다.

증명의 핵심은 크게 3가지 인데, (1) $m_{\mathcal{H}}$를 어떻게 대입할 것인가, (2) $E_{out}$은 어떻게 되는가, (3) 그리고 이를 합치는 과정입니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828746-d0a83f2c-2edf-11e8-9c60-80187044a55b.png"></center>
<br>

먼저 Hoeffding's Inequality에서 $M$이 어떻게 나왔는지와 이를 어떻게 바꾸는지에 대한 대략적인 그림입니다.

가운데 그림이 바로 Hoeffding's Inequality에서 $M$이 곱해진 이유인데, 각각의 Bad Event를 모두 배반사건으로 가정하고 Union Bound를 씌웠기 때문입니다.

하지만 직관적으로 생각해 보았을 때, 이 Bad Event들이 모두 배반사건일리는 없겠죠. 아마 맨 오른쪽 처럼 대부분의 Event가 겹치게 될 텐데, 이는 다음 단원인 VC Dimension에서 다루게 될 예정입니다.

지금은 그냥 저런식으로 해결되겠구나~ 라고만 생각하시면 되겠습니다.

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828757-d84a13fe-2edf-11e8-8c5f-a902890bb394.png"></center>
<br>

슬라이드 18

<br>
<center><img src="https://user-images.githubusercontent.com/35926730/37828762-dca0e28e-2edf-11e8-8940-37dd607310f0.png"></center>
<br>

마지막에서는 이를 모두 합쳐서 최종적으로 변한 식입니다. 이 식을 The Vapnic-Chervonenkis (VC) Inequality라고 합니다.

보시면 식에서 $M$이 $m_{\mathcal{H}}$가 교체된 것 외에도 이것저것 바뀌었음을 알 수 있습니다. 가령 N이 2N으로, $m_{\mathcal{H}}$ 앞의 계수가 2에서 4로 바뀌었고, 지수에서 2가 8분의 1로 바뀌었죠.

왜 이렇게 바뀌었는지 궁금하시면 책 부록에 첨부된 증명을 보시면 될 것 같습니다. (-_-) 사실 저도 증명을 읽어보지 않아서 왜인지 이유는 잘 모르겠습니다만, 그냥 이렇게 바뀌는구나 라고만 이해하시고 그것보다 중요한 VC Bound에 더 중점을 둬야 할 것 같습니다.

이어지는 VC Dimension 단원에서 이 VC Bound에 대해 더 자세히 다뤄질 예정입니다.

이번 단원은 여기까지입니다. 감사합니다.

<br><br>

|<b><center>저도 배우고 있는 학생이라 잘못 이해한 부분이 있을 수 있습니다.</center></b>|
|<b><center>질문이나 틀린 부분에 대한 지적을 댓글로 달아주시면 최대한 빨리 답변해드리겠습니다.</center></b>|