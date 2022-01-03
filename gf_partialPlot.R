gf_partialPlot <- function(model, df, x.var, which.class = NULL){
  # model should be a caret object
  # df is a data frame or tbl
  # x.var is a quoted name of a column in df
  # which.class is a quoted name of a level of the response variable.  This specifies which class we want to count as "1" (numerator of the log odds).  If omitted, uses the 0th class (for coding simplicity and consistency with randomforest::partialPlot).  This argument does nothing for regression problems.
  
  library(ggformula)
  
  observed_data = df[x.var]
  test_vals = seq(min(observed_data), max(observed_data),length = 50)
  pred_avg = numeric(50)
  
  if(model$modelType == "Classification"){
    for(ii in 1:50){
      comp_df <- df 
      comp_df[x.var] = test_vals[ii]
      
      probs = predict(model, comp_df, type = "prob")
      if(is.null(which.class)){
        pred_avg[ii] = mean(log(probs[,1]/(1-probs[,1])))
      }
      else{
        pred_avg[ii] = mean(log(probs[,which.class]/(1-probs[,which.class])))
      }
    } # end iteration over the test_vals
  } # end "if Classification"
  else{ # Regression model
    for(ii in 1:50){
      comp_df <- df 
      comp_df[x.var] = test_vals[ii]
      
      pred_avg[ii] = mean(predict(model, comp_df))
    }
  } #end "if Regression"
  to_return = gf_line(pred_avg ~ test_vals) %>%
    gf_labs(title = paste("Partial dependence on", x.var))
  return(to_return)
} # end of function