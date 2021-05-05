
#' Projection of L2 norm
#'
#' @param x
#' @param tau
#'
#' @return
#' @export
#'
#' @examples

proj_L2_m = function(x, tau){
  lv = sqrt(apply(x^2,2,sum))
  px = x
  temp = matrix(c(tau/lv), dim(x)[1], dim(x)[2], byrow = TRUE) * x
  px[,(lv > tau)] = temp[, (lv > tau)]
  return(px)
}

#' Proximal of L2
#'
#' @param x
#' @param tau
#'
#' @return
#' @export
#'
#' @examples
prox_L2_m = function(x, tau){
  temp1 <- 1 - tau/sqrt(apply(x^2,2,sum))
  temp1 <- ifelse(temp1 < 0, 0, temp1)
  px <- matrix(temp1,dim(x)[1],dim(x)[2],byrow = TRUE) * x
  return(px)
}

#' Projection of infinity norm
#'
#' @param x
#' @param tau
#'
#' @return
#' @export
#'
#' @examples
proj_Linf_m = function(x, tau){
  ax = abs(x)
  temp = matrix(tau, dim(ax)[1], dim(ax)[2], byrow = TRUE)
  ax[which(ax >= temp)] = temp[which(ax >= temp)]
  px = sign.c(ax, x)
  return(px)
}

prox_L1_m = function(x, tau){
  tau = matrix(tau, dim(x)[1], dim(x)[2], byrow = TRUE)
  px = abs(x) - tau
  px[which(px < 0)] = 0
  px = sign.c(px, x)
  return(px)
}

project_to_simplex = function(x, z){
  n = length(x)
  mu = sort(x,  decreasing = TRUE)
  print(mu)
  cumsum = mu[1]
  for(j in 2:n){
    cumsum = cumsum + mu[j]
    if((j * mu[j] - cumsum + z) <= 0){
      rho = j-1
      # print(rho)
      break
    }
    rho = j
  }
  theta = (sum(mu[1:rho]) - z) / rho
  x = ifelse((x - theta) < 0, 0, x-theta)
  return(x)
}

proj_L1_m = function(x, tau){
  px = abs(x)
  for(i in 1:dim(px)[2]){
    px[,i] = project_to_simplex(px[,i], tau[i])
  }
  px = sign.c(px, x)
  return(px)
}

prox_Linf_m = function(x, tau){
  # This function performs the proximal mapping of tau * L-infinity norm.
  # It is computed via Moreau decomposition and Projection onto
  # an L1-ball of radius tau.
  # px = x - project_L1(x,tau)
  px = matrix(1/tau, dim(x)[1], dim(x)[2], byrow = TRUE) * x
  px = x - proj_L1_m(x, tau)
  return(px)
}


sign.c = function(px, x){
  a = ifelse(px == abs(x), x, ifelse(x > 0, abs(px), -abs(px)))
  return(a)
}
