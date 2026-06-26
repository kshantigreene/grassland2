

#Remove the duplicate rows- merge the amp/power columns into one row

#Separate the file path into different columns

#Remove the Sensor text and change to number

#Parse dates and times from file path
parse_calls <- function(calls) {
  calls %>%
    filter(row_number() %% 2 == 0) %>%
    mutate(
      sensor = as.integer(str_extract(begin_path, "(?<=Sensor )\\d+")),
      date   = as.Date(str_extract(begin_path, "\\d{8}(?=\\\\)"), format = "%Y%m%d"),
      time   = format(strptime(str_extract(begin_path, "(?<=_)\\d{6}(?=\\.)"), "%H%M%S"), "%H:%M:%S")
    ) %>%
    select(-view, -channel)
}

analyze_calls <- function(calls, species){

  # Change power to be a positive number (high is most power). 
  #Add the min+1.0 to the power so no 0s
  minPow <-  min(calls$inband_power_d_b_fs)
  calls_adjusted <- calls %>% 
    mutate(power =inband_power_d_b_fs-minPow+1.0)
  
  # arrange by power
  results <-calls_adjusted %>% 
    group_by(sensor) %>%
    summarize(totalPower = sum(power)) %>%
    arrange(sensor)
  
  #get the nest locations
  library(janitor)
  nest_locations <- read_csv("data/nest_locations.csv") %>% clean_names() %>% select(nest_id,species_code)
  sensor_distances=read_csv("data/nest_sensor_distances.csv")%>%
    clean_names() %>% left_join(nest_locations) %>%
    mutate(sensor_id = as.factor(sensor_id)) %>% 
    filter(species_code==species)
  
  
  #compute sum and avg distances to nests
  sensor_avg_dist <- sensor_distances %>% 
    group_by(sensor_id) %>%
    summarize(sum_dist=sum(distance), avg_dist=mean(distance))
  
  #calculate number of days recording per sensor and join with sensor summaries

  results_new <- calls_adjusted %>%
    group_by(sensor,date) %>%
    group_by(sensor) %>%
    count(date) %>%
    group_by(sensor) %>%
    count() %>%
    
    right_join(results) %>%
    rename(num_days=n)
  
  #Add rows for sensors with power of 0
  
  x <- tibble(
    sensor=c(1,2,6,7,10,11,13,25,26),
    totalPower=c(0,0,0,0,0,0,0,0,0),
    num_days=c(0,0,0,0,0,0,0,0,0)
  )
  sensor_data <- bind_rows(results_new, x)
  
  sensor_locations <- read_csv("data/sensor_locations.csv", 
                               col_select=c("sensor","lat","lon"), 
                               col_names = c("sensor","lat","lon"),
                               col_types = cols(sensor=col_factor(),lat = col_double(),
                                                lon=col_double()))
  
  #Turn the sensor into a factor
  sensor_data <- sensor_data %>%
    mutate(sensor=fct_relevel(as.character(sensor),"1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26")) %>% 
    left_join(sensor_locations) %>%
    filter(num_days > 0)
  
  #Compute power average per day (because they all have different numbers of days)
  sensor_data2 <- sensor_data %>%
    mutate(avg_power = totalPower/num_days) %>%
    left_join(sensor_avg_dist,by = c("sensor" = "sensor_id"))
  
  sensor_data2
}
