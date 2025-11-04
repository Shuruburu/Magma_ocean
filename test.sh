#!\bin\bash
keys=($(python3 /home/shurubura/Documents/project/Magma/main.py get_the_keys))
for key in "${keys[@]}" ; do
	# Edditing the python file and the running the simulation
	python3  /home/shurubura/Documents/project/Magma/main.py $key
	python3 /home/shurubura/Documents/VULCAN/vulcan.py
done 


