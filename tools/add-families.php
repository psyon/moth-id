<?php

$species = array();

$fd = fopen('specimen.csv', 'r');
while(!feof($fd)) {
	$row = fgetcsv($fd);
	if(empty($row[0])) continue;

	$species[$row[5]] = $row[3] . ' ' . $row[5];
}
fclose($fd);

$fd = fopen('mpg-families.csv', 'r');
while(!feof($fd)) {
	$row = fgetcsv($fd);
	if(empty($row[0])) continue;

	$species[$row[1]] = $row[0] . ' ' . $row[1];
}
fclose($fd);

$fd = fopen('species.txt', 'r');
while(!feof($fd)) {
	$line = trim(fgets($fd));
	if(empty($line)) continue;

	echo $species[$line] . "\n";
}
fclose($fd);
