library(tidyverse)
library(lme4)

# Unlike the Python implementation in empirical_bayes.py, this 
# won't deal nicely with groups of size n=1. If you have such groups
# in your data, you should probably add some edge case logic.

# Create some data with unequal group sizes.
df <- rbind(head(filter(iris, Species == 'setosa'), 5),
            head(filter(iris, Species == 'versicolor'), 7),
            head(filter(iris, Species == 'virginica'), 4)) %>%
  select(Species, Sepal.Width)

# Multi Sample Size James-Stein Estimator
mss_js <- df %>%
  group_by(Species) %>%
  summarise(
    x_i = mean(Sepal.Width),
    epsilon2_i_hat = var(Sepal.Width),
    n = n(),
    sigma2_i_hat = epsilon2_i_hat / n) %>%
  mutate(
    X_bar = mean(x_i),
    btw_group_std2 = var(x_i),
    B_hat = sigma2_i_hat / btw_group_std2,
    B_hat = pmin(pmax(B_hat, 0), 1),
    x_i_hat = (1 - B_hat) * x_i + B_hat * X_bar)

# Multi Sample Size Pooled James-Stein Estimator
mss_js_pooled <- df %>%
  group_by(Species) %>%
  summarise(
    x_i = mean(Sepal.Width),
    raw_epsilon2_i_hat = var(Sepal.Width),
    n = n()) %>%
  mutate(
    X_bar = mean(x_i),
    btw_group_std2 = var(x_i),
    epsilon2_hat = sum((n - 1) * raw_epsilon2_i_hat) / sum(n - 1),
    sigma2_hat = epsilon2_hat / n,
    B_hat = sigma2_hat / btw_group_std2,
    x_i_hat = (1 - B_hat) * x_i + B_hat * X_bar)
         
# Mixed Model
lmm <- lmer(Sepal.Width ~ (1 | Species), data = df)
predict(lmm)
