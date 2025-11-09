#    python3 /home/shurubura/Documents/VULCAN/vulcan.py
#!/bin/bash

# Capture the keys from the Python script
keys=($(python3 /home/shurubura/Documents/project/Magma/main.py get_the_keys))
# Loop through each key and run the necessary Python scripts
#echo "the ${keys[@]}"
current=$(pwd)

cd /home/shurubura/Documents/VULCAN  

for key in "${keys[@]}" ; do
    # Running main.py with Species and the key
    python3 /home/shurubura/Documents/project/Magma/main.py Species "$key"

    # Running vulcan.py
    python3 vulcan.py
done 
cd $current

