library(combinat)
library(ggplot2)
library(gridExtra)

#############
# Functions #
#############

APcrossings <- function(N_AP) {   # make the appropriate crossings
  # First N_AP crossings
  APs.cross = sapply(2:N_AP, function(x){
    c(1,x)
  })
  APs.cross = t(APs.cross)
  
  # Creating rest of the crossings
  for (i in 2:(N_AP-1))
    for (j in (i+1):N_AP)
      APs.cross = rbind(APs.cross, c(i,j))
  return (APs.cross)
}

globalAngle <- function(APs, mobile) {   # calculate global angles
  sapply(1:nrow(APs), function(i){
    deltay = mobile$y-APs$y[i]
    deltax = mobile$x-APs$x[i]  
    global_angle = atan(deltax/deltay)*180/pi
    if (deltay < 0)
      global_angle = 180 + global_angle
    if (deltay > 0 & deltax < 0)
      global_angle = 360 + global_angle
    global_angle
  })
}

localAngle <- function(APs) {   # calculate local angles
  sapply(1:nrow(APs), function(i){
    angle = APs$direction[i] - APs$global.angle[i]
    if (angle > 180)
      angle = angle - 360
    if (angle < -180)
      angle = angle + 360
    angle
  })
}

nonSat <- function(APs, APs.cross) {   # remove sasturated APs
  # Remove saturations 
  APs.cross.sat = which(APs$mobile.sat)
  # Remove saturations 
  APs.cross.sat = which(APs$mobile.sat)
  if (length(APs.cross.sat) > 0)
    APs.cross.nonsat = subset(APs.cross, (APs.cross[,1] != APs.cross.sat & APs.cross[,2] != APs.cross.sat))
  else 
    APs.cross.nonsat = APs.cross
  return (APs.cross.nonsat)
}

createCrossings <- function(APs, APs.cross) {   # find crossings
  APs.cross.points = data.frame( x = rep(1,nrow(APs.cross)), y = rep(1,nrow(APs.cross)))
  APs.cross.points$x = sapply(1:nrow(APs.cross), function(i){
    (APs$y.intercept[APs.cross[i, 2]] - APs$y.intercept[APs.cross[i, 1]])/
      (APs$predicted.slope[APs.cross[i, 1]] - APs$predicted.slope[APs.cross[i, 2]])
  })
  
  APs.cross.points$y = sapply(1:nrow(APs.cross), function(i){
    APs$predicted.slope[APs.cross[i, 1]]*(APs$y.intercept[APs.cross[i, 2]] - APs$y.intercept[APs.cross[i, 1]])/
      (APs$predicted.slope[APs.cross[i, 1]] - APs$predicted.slope[APs.cross[i, 2]]) + APs$y.intercept[APs.cross[i, 1]]
  })
  return (APs.cross.points)
}

euc <- function(point1, point2) {   # calculate euclidian distance
  distance = sqrt((point1[1] - point2[1])^2 + (point1[2] - point2[2])^2)
  return(distance)
}

CrossingsCluster <- function(APs.cross) {   # find crossings
  # predict location based on weighted average based on total SD
  position.predict.x = weighted.mean(APs.cross$x, APs.cross$weights)
  position.predict.y = weighted.mean(APs.cross$y, APs.cross$weights)
  
  dists = sqrt((APs.cross$x - position.predict.x)^2 + (APs.cross$y - position.predict.y)^2)
  mean_distance =  mean(dists)
  
  if (nrow(subset(APs.cross, dists < mean_distance * N_dists)) > 0)
    APs.cross = subset(APs.cross, dists < mean_distance * N_dists)
  position.predict.x = weighted.mean(APs.cross$x, APs.cross$weights)
  position.predict.y = weighted.mean(APs.cross$y, APs.cross$weights)
  
  # returns the meausurement
  return (c(position.predict.x, position.predict.y))
}


removeOutside <- function(position.predict) {   # remove samples from outside
  if (position.predict[1] < 0)
    position.predict[1] = 0
  if (position.predict[1] > x_max)
    position.predict[1] = x_max
  if (position.predict[2] < 0)
    position.predict[2] = 0
  if (position.predict[2] > y_max)
    position.predict[2] = y_max
  position.predict
}

SDcalc_v2 <- function(SD1, SD2, angle) {   # calculate maximum SD
  # Calculate SDmax
  SDmax = sapply(1:length(SD1), function(i){
    # Calculate SD(angle)
    sigma.angle = sapply(seq(0,180,5), function(x){
      sigmai = 1 / (abs(cos(x * pi / 180) / SD1[i]) + 
                      abs(cos((x + angle[i]) * pi / 180) / SD2[i]))
    })
    max(sigma.angle)
  })
  return(SDmax)
}

SDcalc_v2.1 <- function(SD1, SD2, angle) {   # calculate maximum SD
  # Calculate SD total
  # Calculate SDmax
  angle_scan = seq(0,180,5)
  # Calculate SD(angle) max
  max_SD_angle = sapply(1:length(angle), function(i){
    angle_max = which.max(1 / (abs(cos(angle_scan * pi / 180) / SD1[i]) + 
                                 abs(cos((angle_scan + angle[i]) * pi / 180) / SD2[i])))
    angle_max
  })
  SDtot = sqrt((1 / (abs(cos(max_SD_angle * pi / 180) / SD1) + abs(cos((max_SD_angle + angle) * pi / 180) / SD2)))^2 +
                 (1 / (abs(cos((max_SD_angle + 90) * pi / 180) / SD1) + abs(cos(((max_SD_angle + 90) + angle) * pi / 180) / SD2)))^2)
  return(SDtot)
}

#############
# Constants #
#############

# Grid dimesions in meters, resolution of 1 meter
x_max = 100
y_max = 100

# Minimum distance
r_sat = 10

# standard error in degrees
Standard.error = 4

# Resolution of samples
res = 10

# Placing and directing APs, direction is the AP main direction. phi = 0 -> y+, phi = 90 -> x+. like a compass
AP1 = data.frame(x = -1, y = -1, direction = 45)
AP2 = data.frame(x = -1, y = 101, direction = 135)
AP3 = data.frame(x = 101, y = 101, direction = 225)
AP4 = data.frame(x = 101, y = -1, direction = 315)
APs = rbind(AP1, AP2, AP3, AP4)
rm(AP1, AP2, AP3, AP4)

# number of repeats
N = 20

# angle limits
min_angle = -65
max_angle = 65

# remove far measurements
N_dists = 2

# exponentual smoothing, alpha index
alpha = 0.3
trend = 0.2

###################################
# Simulate mobile-station's track #
###################################

# straight line (5,5) -> (50, 50)
duration1 = 16
track1= data.frame(rep(0,duration1))
track1$x = sapply(1:duration1, function(x){
  (5 + 3*(x-1))
})
track1$y = sapply(1:duration1, function(x){
  (5 + 3*(x-1))
})
track1$rep.0..duration1. = NULL

# straight line (50, 50) -> (14, 50)
duration2 = 12
track2= data.frame(rep(0,duration2))
track2$x = sapply(1:duration2, function(x){
  (50 - 3*x)
})
track2$y = sapply(1:duration2, function(x){
  50
})
track2$rep.0..duration2. = NULL

# half circle clockwise (14, 50) -> (86, 50)
duration3 = 30
track3= data.frame(rep(0,duration3))
track3$x = sapply(1:duration3, function(x){
  (50 - 36*cos(pi*x/duration3))
})
track3$y = sapply(1:duration3, function(x){
  (50 + 36*sin(pi*x/duration3))
})
track3$rep.0..duration3. = NULL

# straight line (86, 50) -> (65, 50)
duration4 = 7
track4= data.frame(rep(0,duration4))
track4$x = sapply(1:duration4, function(x){
  (86 - 3*x)
})
track4$y = sapply(1:duration4, function(x){
  50
})
track4$rep.0..duration4. = NULL

# half circle clockwise (65, 50) -> (35, 50)
duration5 = 12
track5= data.frame(rep(0,duration5))
track5$x = sapply(1:duration5, function(x){
  (50 + 15*cos(pi*x/duration5))
})
track5$y = sapply(1:duration5, function(x){
  (50 - 15*sin(pi*x/duration5))
})
track5$rep.0..duration5. = NULL

track = rbind(track1, track2, track3, track4, track5)

duration = duration1 + duration2 + duration3 + duration4 + duration5

####################
# Start simulation #
####################

# creating crossings list
N_AP = nrow(APs)
APs.cross = APcrossings(N_AP)

# Use raw positioning
track_position = sapply(1:duration, function(i) {
  # mobile
  mobile = data.frame(x = track$x[i], y = track$y[i])
  
  # calculate distance
  APs$mobile.dist = sqrt((mobile$x - APs$x) ^ 2 + (mobile$y - APs$y) ^
                           2)
  
  # decides if in saturated
  APs$mobile.sat = APs$mobile.dist <= r_sat
  
  # Creating global angle matrix for APs
  APs$global.angle = globalAngle(APs, mobile)
  
  # Creating local angle matrix for APs
  APs$local.angle = localAngle(APs)
  
  # Simulating APs local angle by normal distribution
  # if the angle is out of -65 to 65 degress, it gives radom angle
  APs$predicted.local.angle = sapply(1:nrow(APs), function(i) {
    if (APs$local.angle[i] > min_angle & APs$local.angle[i] < max_angle)
      rnorm(1, APs$local.angle[i], Standard.error)
    else
      runif(1, min_angle, max_angle)
  })
  
  # Converting to globally predicted angle
  APs$predicted.global.angle = APs$direction - APs$predicted.local.angle
  
  # Converting globally predicted angle to a slope
  # Need to build a special case for 0,180 degrees angle... x = x_ap
  APs$predicted.slope = sapply(1:nrow(APs), function(i) {
    (1 / tan(APs$predicted.global.angle[i] / 180 * pi))
  })
  
  # lines intercept
  APs$y.intercept = APs$y - APs$predicted.slope * (APs$x)
  
  # Checking saturation
  APs.cross.nonsat = nonSat(APs, APs.cross)
  
  # Calculating cross-section
  APs.cross.points = createCrossings(APs, APs.cross.nonsat)
  
  # Calculate distance between APs and cross point
  APs.cross.points$AP1 = APs.cross.nonsat[,1]
  APs.cross.points$AP2 = APs.cross.nonsat[,2]
  # Find distance between the estimated intersection to the APs
  APs.cross.points$r1 = sapply(1:nrow(APs.cross.points), function(i) {
    euc(c(APs$x[APs.cross[i,1]], APs$y[APs.cross[i,1]]), c(APs.cross.points$x[i], APs.cross.points$y[i]))
  })
  APs.cross.points$r2 = sapply(1:nrow(APs.cross.points), function(i) {
    euc(c(APs$x[APs.cross[i,2]], APs$y[APs.cross[i,2]]), c(APs.cross.points$x[i], APs.cross.points$y[i]))
  })
  # Find angle between the estimated intersection to the APs
  APs.cross.points$Angle1 = sapply(1:nrow(APs.cross.points), function(i) {
    APs$predicted.global.angle[APs.cross.points$AP1[i]]
  })
  APs.cross.points$Angle2 = sapply(1:nrow(APs.cross.points), function(i) {
    APs$predicted.global.angle[APs.cross.points$AP2[i]]
  })
  
  # Find the SD for each AP
  APs.cross.points$SD1 = APs.cross.points$r1 * sin(Standard.error * pi / 180)
  APs.cross.points$SD2 = APs.cross.points$r2 * sin(Standard.error * pi / 180)
  
  APs.cross.points$AngleDif = sapply(1:nrow(APs.cross.points), function(i) {
    angle1 = APs.cross.points$Angle1[i]
    angle2 = APs.cross.points$Angle2[i]
    if (angle1 > 180)
      angle1 = angle1 - 180
    if (angle2 > 180)
      angle2 = angle2 - 180
    min(abs(angle1 - angle2), abs(180 - abs(angle1 - angle2)))
  })
  
  # Calculate total SD
  APs.cross.points$SDmax = SDcalc_v2(APs.cross.points$SD1, APs.cross.points$SD2, APs.cross.points$AngleDif)
  
  # Calculate weights
  APs.cross.points$weights = 1 / APs.cross.points$SDmax
  
  # Remove really bad crossings
  APs.cross.points = subset(APs.cross.points, APs.cross.points$x > -x_max/2 & APs.cross.points$x < 3*x_max/2)
  APs.cross.points = subset(APs.cross.points, APs.cross.points$y > -y_max/2 & APs.cross.points$y < 3*y_max/2)
  
  # cluster technique
  position.predict = CrossingsCluster(APs.cross.points)
  
  if (is.nan(position.predict[1])) {
    print(position.predict)
    print(APs.cross.points)
    print(mobile)
  }
  
  position.predict = removeOutside(position.predict)
  c(position.predict)
})
track_position = as.data.frame(t(track_position))
colnames(track_position) = c("x", "y")

# applying exponential smoothing
exp_pos = data.frame(Lx = rep(0, duration), Ly = rep(0, duration), Tx = rep(0, duration), Ty = rep(0, duration))

# Get the aproximated position
exp_pos$Lx[1] = track_position$x[1]
exp_pos$Ly[1] = track_position$y[1]

# Get the aproximated trend
exp_pos$Tx = 0
exp_pos$Ty = 0

for (i in seq(2,duration)){
  exp_pos$Lx[i] = (1 - alpha) * (exp_pos$Lx[i-1] + exp_pos$Tx[i-1]) + alpha * track_position$x[i]
  exp_pos$Tx[i] = trend * (exp_pos$Lx[i] - exp_pos$Lx[i-1]) + (1 - trend) * exp_pos$Tx[i-1]
  
  exp_pos$Ly[i] = (1 - alpha) * (exp_pos$Ly[i-1] + exp_pos$Ty[i-1]) + alpha * track_position$y[i]
  exp_pos$Ty[i] = trend * (exp_pos$Ly[i] - exp_pos$Ly[i-1]) + (1 - trend) * exp_pos$Ty[i-1]
}

exp_SSE = sum((exp_pos$Lx-track$x)^2+(exp_pos$Ly-track$y)^2)
RSME = sqrt(exp_SSE/nrow(exp_pos))

# Plot
mobilex = data.frame(t = seq(duration))
mobilex$truex = track$x
mobilex$posx = track_position$x
mobilex$expx = exp_pos$Lx

mobiley = data.frame(t = seq(duration))
mobiley$truey = track$y
mobiley$posy = track_position$y
mobiley$expy = exp_pos$Ly

# split data to x and y
p1 = 
  ggplot(mobilex, aes(x = t)) +
  geom_line(aes(y = truex), colour="red") +
  geom_line(aes(y = posx), colour="blue") +
  geom_line(aes(y = expx), colour="green", size = 1.25) +
  scale_x_continuous(limits=c(0, duration)) +
  scale_y_continuous(limits=c(0, x_max)) +
  xlab("t") + ylab("x") + ggtitle("Positioning x")

p2 = 
  ggplot(mobiley, aes(x = t)) +
  geom_line(aes(y = truey), colour="red") +
  geom_line(aes(y = posy), colour="blue") +
  geom_line(aes(y = expy), colour="green", size = 1.25) +
  scale_x_continuous(limits=c(0, duration)) +
  scale_y_continuous(limits=c(0, y_max)) +
  xlab("t") + ylab("y") + ggtitle("Positioning y")

grid.arrange(p1, p2, ncol=1)

# show in 2D
true_pos = data.frame(x = track$x, y = track$y)
raw_pos = data.frame(x = track_position$x, y = track_position$y)
exp_smooth_pos = data.frame(x = exp_pos$Lx, y = exp_pos$Ly)
p3 = 
  ggplot(true_pos, aes(x = x)) +
  geom_path(aes(y = y), colour="red") +
  geom_path(data = raw_pos, aes(x = x, y = y), colour="blue", size = (1 + 0.25*(mobilex$t)/duration)) +
  geom_path(data = exp_smooth_pos, aes(x = x, y = y), colour="green", size = (1 + 1*(mobilex$t)/duration)) +
  scale_x_continuous(limits=c(0, x_max)) +
  scale_y_continuous(limits=c(0, y_max)) +
  xlab("x") + ylab("y") + ggtitle("Positioning")
p3

RSME