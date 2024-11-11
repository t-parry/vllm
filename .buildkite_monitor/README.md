Running the script requires BUILDKITE_API_TOKEN, request it from [here](https://buildkite.com/user/api-access-tokens).  
Create .env file in .buildkite_monitor directory and paste there:
> export BUILDKITE_API_TOKEN='YOUR_TOKEN' 

Running the script also requires having GMAIL_USERNAME and GMAIL_PASSWORD in .env.

Use py_3.9 conda environment for development.

One option for a scheduled run would be to use cron:\
On some machine, edit your crontab file:\
`crontab -e`\
Using nano/vim/emacs to add an entry for running the container, e.g.\
`*/15 * * * * docker run --network=host --rm -v $HOME/:/mnt/home/ --name=hissu-bkmonitor hissu-bkmonitor` # Runs the hissu-bkmonitor container every 15 minutes    

To cancel or change the cron job:  

`crontab -e`\
Comment out or change the line with the job, then  
`service cron reload`


As a temporary solution we set up saving alerts data in $HOME directory, this directory should be mounted when container is launched (with -v flag). `monitor.py` script has PATH_TO_LOGS set to '/mnt/home/buildkite_logs/', where the logs are saved.

TODO:  
1. ~~Export notebook code to script. - Olga, done~~
2. ~~Function for waiting time calculation, threshold for alert. - Olga, done~~
3. How to not process records that were already processed: store builds with all AMD tests already finished in csv/database, ignore those. - Olga
4. ~~Sending the alert function: email and/or teams. - Hissu, done~~
5. ~~Scheduled start of the script, every 15 mins. - Hissu~~
6. ~~Agent health: for now implement from builds info. - Olga, done~~
7. ~~Add try-except for SMTP authentication fails. - Hissu~~
8. ~~Pass gmail user and password as env variables instead of txt file -Hissu~~
9. ~~Make a Docker container for the monitor script -Hissu~~
10. ~~Parametrize email for alerts, enable sending to multiple addresses. - Olga, done~~
11. Run script as daemon so that if a server reboots, the script will be launched again at startup. - Hissu
12. Job waiting time when it is not yet started, search for job state == waiting. - Olga
