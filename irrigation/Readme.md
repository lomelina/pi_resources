To add something to cron:
crontab -e

Script executing every day 18:30 (use full path):

30 18 * * * /path/to/your/script/script.sh

Two times a week: 
30 18 * * MON,THU /path/to/script.sh
or
30 18 * * 1,4 /path/to/script.sh