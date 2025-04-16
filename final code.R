
# Load Libraries

install.packages("caret")
install.packages("ggplot2")
install.packages("car")
install.packages("lmtest")
install.packages("performance")
install.packages("reshape2")

library(tidyverse)
library(caret)
library(car)
library(ggplot2)
library(lmtest)
library(performance)
library(reshape2)


# Load and Clean Dataset

dataset <- read.csv("C:/Users/farha/OneDrive/Desktop/BAYES/Year 3/Final Year Project/data.csv", 
               fileEncoding = "UTF-8", stringsAsFactors = FALSE)

# Clean column names
colnames(dataset) <- make.names(colnames(dataset))  

# Remove commas and convert to numeric
dataset$Attendance <- as.numeric(gsub(",", "", dataset$Attendance))
dataset$Stadium.Capacity <- as.numeric(gsub(",", "", dataset$Stadium.Capacity))


# Exploratory Data Analysis Before Split

# Basic structure and missing values
str(dataset)
summary(dataset)
colSums(is.na(dataset))

# Extract hour from Time
dataset$Hour <- as.numeric(substr(dataset$Time, 1, 2))

# Group kickoff times
dataset$Time.Category <- cut(dataset$Hour,
                        breaks = c(0, 13, 17, 24),
                        labels = c("Early", "Afternoon", "Evening"),
                        include.lowest = TRUE)

# Boxplot: Attendance by Time Category
ggplot(dataset, aes(x = Time.Category, y = Attendance)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Attendance by Kickoff Time")

# Game Importance
dataset$Game.Importance <- ifelse(dataset$Importance.of.the.game %in% c("Championship Final", 
                                                              "Championship Semi-final", 
                                                              "Opening weekend",
                                                              "Final Matchday before playoffs", 
                                                              "Relegation battle", 
                                                              "Rivalry"), "High",
                             ifelse(dataset$Importance.of.the.game %in% c("Early May Bank Holiday",
                                                                     "Good Friday (Bank Holiday)",
                                                                     "Easter Monday (Bank Holiday)",
                                                                     "Christmas Day",
                                                                     "Bank holiday for the coronation of King Charles III",
                                                                     "New Year's Day",
                                                                     "First home game", 
                                                                     "Boxing Day"), "Medium", "Low"))
dataset$Game.Importance <- factor(dataset$Game.Importance, 
                             levels = c("High", "Medium", "Low"))


# Boxplot: Attendance by Game Importance
ggplot(dataset, aes(x = Game.Importance, y = Attendance)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "Attendance by Game Importance")

# Correlation heatmap
numeric_vars <- dataset %>% select(where(is.numeric)) %>% na.omit()
cor_matrix <- cor(numeric_vars)
cor_melted <- melt(cor_matrix)

ggplot(cor_melted, aes(Var1, Var2, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "red", high = "green", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1)) +
  labs(title = "Correlation Heatmap of Numeric Variables",
       x = "", y = "")


# Train/Test Split
set.seed(123)
trainIndex <- createDataPartition(dataset$Attendance, p = 0.8, list = FALSE)
train <- dataset[trainIndex, ]
test <- dataset[-trainIndex, ]

# Feature Engineering Function
feature_engineering <- function(data) {
  data$Hour <- as.numeric(substr(data$Time, 1, 2))
  data$Time.Category <- cut(data$Hour,
                            breaks = c(0, 13, 17, 24),
                            labels = c("Early", "Afternoon", "Evening"),
                            include.lowest = TRUE)
  data$Game.Importance <- ifelse(data$Importance.of.the.game %in% c("Championship Final", 
                                                                    "Championship Semi-final", 
                                                                    "Opening weekend",
                                                                    "Final Matchday before playoffs", 
                                                                    "Relegation battle", 
                                                                    "Rivalry"), "High",
                                 ifelse(data$Importance.of.the.game %in% c("Early May Bank Holiday",
                                                                           "Good Friday (Bank Holiday)",
                                                                           "Easter Monday (Bank Holiday)",
                                                                           "Christmas Day",
                                                                           "Bank holiday for the coronation of King Charles III",
                                                                           "New Year's Day",
                                                                           "First home game", 
                                                                           "Boxing Day"), "Medium", "Low"))
  data$Game.Importance <- as.factor(data$Game.Importance)
  data$Time.Category <- as.factor(data$Time.Category)
  data$Day3 <- as.factor(data$Day3)
  return(data)
}

train <- feature_engineering(train)
test <- feature_engineering(test)

# Set Low as base level for Game Importance
train$Game.Importance <- relevel(train$Game.Importance, ref = "Low")
test$Game.Importance <- relevel(test$Game.Importance, ref = "Low")

# Set Afternoon as base level for Time Category
train$Time.Category <- relevel(train$Time.Category, ref = "Afternoon")
test$Time.Category <- relevel(test$Time.Category, ref = "Afternoon")

# Build Regression Model
model <- lm(Attendance ~ Stadium.Capacity + Game.Importance + 
              Distance.travelled.by.away.supporters..miles. + 
              Time.Category + Day3, data = train)

summary(model)
vif(model)

model_interaction1 <- lm(Attendance ~ Stadium.Capacity + Game.Importance + 
                           Distance.travelled.by.away.supporters..miles. * Time.Category + 
                           Day3, data = train)

summary(model_interaction1)

model_interaction2 <- lm(Attendance ~ Stadium.Capacity + Game.Importance + 
                           Distance.travelled.by.away.supporters..miles. * Day3 + 
                           Time.Category, data = train)

summary(model_interaction2)


# Diagnostics
par(mfrow = c(2, 2))
plot(model)
par(mfrow = c(1, 1))
qqnorm(residuals(model))
qqline(residuals(model), col = "blue")


# Model Evaluation on Test Set
predictions <- predict(model, newdata = test)
RMSE <- sqrt(mean((test$Attendance - predictions)^2))
cat("RMSE on Test Set:", RMSE, "\n")


# Visualizations
# Predicted vs Actual
model_data_test <- test
model_data_test$predicted <- predictions

ggplot(model_data_test, aes(x = predicted, y = Attendance)) +
  geom_point(color = "steelblue", alpha = 0.7) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Predicted vs Actual Attendance (Test Set)", 
       x = "Predicted", y = "Actual")

# Residual Plot
ggplot(model_data_test, aes(x = predicted, y = Attendance - predicted)) +
  geom_point(color = "red") +
  geom_hline(yintercept = 0, linetype = "dashed") +
  labs(title = "Residual Plot", x = "Predicted Attendance", y = "Residuals")

model_data_test$error <- abs(model_data_test$Attendance - model_data_test$predicted)

# View largest errors
head(model_data_test[order(-model_data_test$error), c("Home", "Away", "Attendance", "predicted", "error")], 10)


