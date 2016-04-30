# Binomal probability training

# The probability of conceiving a girl is 0.49. 
# What is the probability that a family with 4 children 
# has 2 girls and 2 boys (you can assume no twins)?
dbinom(2, 4, prob = 0.49)

# What is the probability that a family with 10 children 
# has 4 girls and 6 boys (you can assume no twins)?
dbinom(4, 10, prob = 0.49)

# The genome has 3 billion bases. 
# About 20% are C, 20% are G, 30% are T and 30% are A. 
# Suppose you take a random interval of 20 bases, 
# what is the probability that the GC-content (proportion of Gs or Cs) 
# is strictly above 0.5 in this interval (you can assume independence)?
1-pbinom(10, 20, prob = 0.4)


# The probability of winning the lottery is 1 in 175,223,510. 
# If 189,000,000 randomly generated (with replacement) tickets are sold,
# what is the probability that at least one winning tickets is sold? 
# (give your answer as a proportion not percentage)
1-pbinom(0, 189000000, prob = (1/175223510))

# What is the probability that two or more winning tickets are sold?
1-pbinom(1, 189000000, prob = (1/175223510))

pbinom(9, 20, prob = 0.4) - pbinom(7, 20, prob = 0.4)

N = 3000000000
p = 0.4
E = N*p
V = N*p*(1-p)
sd = sqrt(V)
pnorm((N*0.45), mean = E, sd = sd) - pnorm((N*0.35), mean = E, sd = sd)

b <- (9 - 20*.4)/sqrt(20*.4*.6)
a <- (7 - 20*.4)/sqrt(20*.4*.6)
pnorm(b)-pnorm(a)

b <- (450 - 1000*.4)/sqrt(1000*.4*.6)
a <- (350 - 1000*.4)/sqrt(1000*.4*.6)
pnorm(b)-pnorm(a)

# difference between binominal to normal distirbutions.
pbinom(450, 1000, prob = 0.4) - pbinom(350, 1000, prob = 0.4) - (pnorm(b)-pnorm(a))

# Normal approximation
# Compare the normal approximation and exact probability (from binomial) 
# of the proportion of Cs being . Plot the exact versus approximate 
# probability for each and combination
Ns <- c(5,10,30,100)
ps <- c(0.01,0.10,0.5,0.9,0.99)
library(rafalib)
mypar2(4,5)
for(N in Ns){
  ks <- 1:(N-1)
  for(p in ps){
    exact = dbinom(ks,N,p)
    a = (ks+0.5 - N*p)/sqrt(N*p*(1-p))
    b = (ks-0.5 - N*p)/sqrt(N*p*(1-p))
    approx = pnorm(a) - pnorm(b)
    LIM <- range(c(approx,exact))
    plot(exact,approx,main=paste("N =",N," p = ",p),xlim=LIM,ylim=LIM,col=1,pch=16)
    abline(0,1)
  }
}


# The normal approx. for binominal probability doesn't work for p~0,1
N <- 189000000
p <- 1/175223510
dbinom(2,N,p)

a <- (2+0.5 - N*p)/sqrt(N*p*(1-p))
b <- (2-0.5 - N*p)/sqrt(N*p*(1-p))
pnorm(a) - pnorm(b)

# What is the Poisson approximation for 
# the probability of two or more person winning?
# The poisson does
1-dpois(0,N*p)-dpois(1,N*p)

###############################
# Maximum Likelihood Estimate #
###############################

library(devtools)
install_github("genomicsclass/dagdata")

library(dagdata)
data(hcmv)

library(rafalib)
mypar2()
plot(locations,rep(1,length(locations)),ylab="",yaxt="n")

breaks=seq(0,4000*round(max(locations)/4000),4000)
tmp=cut(locations,breaks)
counts=as.numeric(table(tmp))
hist(counts)

probs <- dpois(counts,4)
likelihood <- prod(probs)
likelihood

logprobs <- dpois(counts,4,log=TRUE)
loglikelihood <- sum(logprobs)
loglikelihood

# Now write a function that takes and the vector of counts as input, 
# and returns the log-likelihood. Compute this log-likelihood for
# lambdas = seq(0,15,len=300) and make a plot.
lambdas = seq(0,15,len=300)

loglikelihood = sapply(1:length(lambdas), function(i){
  logprobs <- dpois(counts,lambdas[i],log=TRUE)
  loglikelihood <- sum(logprobs)
  loglikelihood
})

which.max(loglikelihood)
lambdas[104]

# The point of collecting this dataset was to try to determine
# if there is a region of the genome that has higher palindrome 
# rate than expected. We can create a plot and see the counts per location:
breaks=seq(0,4000*round(max(locations)/4000),4000)
tmp=cut(locations,breaks)
counts=as.numeric(table(tmp))
binLocation=(breaks[-1]+breaks[-length(breaks)])/2
plot(binLocation,counts,type="l",xlab=)

which.max(counts)
binLocation[24]
counts[24]

lambda = mean(counts[ - which.max(counts) ])