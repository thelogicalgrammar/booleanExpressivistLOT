while read p; do
	echo "$p"
	# timeout 90m python3 utilities.py --primitives $p --action "single_minimal"
	python3 utilities.py --primitives $p --action "single_minimal"
done < functionally_complete.txt
