while read p; do
	echo "$p"
	timeout 5m python3 utilities.py --primitives $p --action "single_minimal"
done < functionally_complete.txt
