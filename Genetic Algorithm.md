#### 遗传算法

遗传算法使用群体搜索技术，将种群代表问题的一组可行解，通过对当前种群施加选择、交叉、变异等一系列遗传操作来产生新一代的种群，并逐步使种群进化到包含近似全局最优解的状态。

optimizer $f(x_{1},...,x_{n}) = output$

简而言之，遗传算法是一种优化算法。

#####一、基本术语：

+ Population：即种群或总体。是所有可行解(candidate solution)的集合。
+ Gene：基因。可行解编码的分量，也就是组成可能解的具体数字
+ Chromosome：染色体。可行解的编码。染色体由基因构成，可行解的编码位数就是基因，就是可行解编码的分量。
+ mutation：变异。基因变异导致染色体的变化。譬如一条染色体为[1,3,5,7]，也就是一组可能解，变为染色体[2,3,4,7]，这里就发生了两个基因变异：1->2，5->4。变异的目的在于，在该可能解的周围随机产生另外一个可能解，这有助于避免陷入局部最优点。
+ crossover：配对。染色体的配对会得到新的染色体后代。也就是两组可能解之间按照某种自由组合规则，产生出另外不一样的解。**变异往往选择发生在配对之后的染色体上**。
+ Selection：选择。适者生存，首先从总体Population中计算出每条Chromosome的适应力，并以此作为筛选的依据。如果某条染色体适应力强，那么它就能繁衍出更多的后代，具体代码体现在对其抽样数的增加。
+ Fitness：适应力。每条染色体都有它自己的fitness值。也就是可能解和目标解之间的距离。距离越小的可能解会存活下来，而距离越大的可能解则被淘汰。

##### 二、算法的参数

遗传算法是可以根据不同问题场景进行定制的，因此，它的种类繁多，但是最基本的参数是不变的，分别是：population size， mutation rate。

族群大小：种群的大小等价于搜索空间的大小。搜索空间越大，越能够找到全局最优解，但计算的时间也就会越长。

变异率：适当的变异率能够增加种群中可能解的多样性，这有助于避免得到局部最优解。变异率过小，会导致族群中的多样性低，使得搜索到最优解的时间加长，而过高的变异率又会造成种群中的多样性太高，就像随机生成的一样，这会导致搜索空间的震荡性，会使得算法不会收敛。如果变异率设置为0，那么解的多样性就只能通过配对crossover得到，这降低了种群中的个体多样性，也增加了算法的运行时间。

#####三、对遗传算法的理解：

+ 从问题中抽象出“基因”和“显性性状”：

  对问题进行解析，抽象出问题的属性，这些属性即可视作“基因”，而生物学里基因决定着显性现状，在这里，问题的“基因”决定问题的最终“显性”。举个风控的例子，每个用户的数据特征作为“基因”，而最终的风控结果，即由那些“基因”决定。

  对从问题中抽象出来的“基因”设计一个DNA类。

+ 构建群体，并初始化。群体由个体组成，每一个个体都是DNA类的实例。

+ 设计选择算法：

  适者生存。定义生存的条件，就是设计一个fitness函数，该函数通过计算每个个体的某个指标作为个体的fitness能力，并筛选出适应的个体。

  配对繁衍。将筛选出的个体放入mating pool “配对池”，并进行随机配对得到后代。

  繁衍规则。定义繁衍规则函数crossover以及基因变异函数mutation。

以上的过程，可以通过伪代码的形式表示如下：

> START
>
> Generate the initial population
>
> Compute fitness
>
> REPEAT
>
> ​	Selection
>
> ​	Crossover
>
> ​	Mutation
>
> ​	Compute fitness
>
> UNTIL population has converged
>
> STOP

#####四

遗传算法的两个主要部分在于：1、如何对所求问题进行编码；2、如何设计fitness评估函数。

#### 遗传算法应用于深度学习模型的训练

> START
>
> Generate N random networks to create population
>
> Compute the fitness of each network
>
>  1.  train the weights of each network
>
>  2.  then see how well it performs at test set. using classification accuracy as
>
>      the fitness function
>
> Sort all the networks in population by accuracy score, and keep some percentage of the top networks to become part of the next generation to generate children.
>
> Also randomly keep a few of the non-top networks. This helps find potentially lucky combinations between worse-performers and top performers, and also helps keep from getting stuck in a local maximum.
>
> Randomly mutate some of the parameters on the retained networks

以上步骤是网上看来的，需要做实验验证。