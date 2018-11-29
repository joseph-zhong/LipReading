#!/bin/bash

echo 'Removing Icon files generated from Google Drive...'
find ./data -type f -name 'Icon?' -print -delete
