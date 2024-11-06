Running the script requires BUILDKITE_API_TOKEN, request it from [here](https://buildkite.com/user/api-access-tokens).  
Create .env file in .buildkite_monitor directory and paste there:
> export BUILDKITE_API_TOKEN='YOUR_TOKEN' 

Running the script also requires having GMAIL_USERNAME and GMAIL_PASSWORD in .env.

Use py_3.9 conda environment for development.

One option for a scheduled run would be to use cron:
On some machine, edit your crontab file:
`crontab -e`
Using nano/vim/emacs to add an entry for running the container, e.g.
`*/15 * * * * docker run --network=host --rm --name=hissu-bkmonitor hissu-bkmonitor` # Runs the hissu-bkmonitor container every 15 minutes
(I'm not setting this up yet, as we don't have a check for previously identified issues, so atm this would spam us about the same incidents repeatedly) 

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
