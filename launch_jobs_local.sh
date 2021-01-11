while read p; do
	echo "$p"
	timeout 30m python3 utilities.py --primitives $p --action "single_minimal"
done < functionally_complete.txt
