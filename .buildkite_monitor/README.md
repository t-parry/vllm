Running the script requires BUILDKITE_API_TOKEN, request it from [here](https://buildkite.com/user/api-access-tokens).  
Create .env file in .buildkite_monitor directory and paste there:
> export BUILDKITE_API_TOKEN='YOUR_TOKEN' 

Use py_3.9 conda environment for development.


TODO:  
1. ~~Export notebook code to script. - Olga, done~~
2. ~~Function for waiting time calculation, threshold for alert. - Olga, done~~
3. How to not process records that were already processed: store builds with all AMD tests already finished in csv, ignore those.
4. ~~Sending the alert function: email and/or teams. - Hissu, done~~
5. Scheduled start of the script. - Hissu
6. ~~Agent health: for now implement from builds info. - Olga, done~~
7. Add try-except if credentials file is not found or if login throws an error.
8. ~~Parametrize email for alerts, enable sending to multiple addresses. - Olga, done~~
9. Run script as daemon so that if a server reboots, the script will be launched again at startup.
10. Job waiting time when it is not yet started.