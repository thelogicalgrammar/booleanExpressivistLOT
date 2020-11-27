while IFS="" read -r p || [ -n "$p" ]
do
	job.sh "$p"
done < functionally_complete.txt
