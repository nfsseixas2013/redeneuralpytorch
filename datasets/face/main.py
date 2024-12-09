import csv

def remove_comments_from_csv(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.reader(infile)
        processed_rows = []
        
        for row in reader:
            processed_row = [
                cell.split('#')[0].strip() if '#' in cell else cell.strip() 
                for cell in row
            ]
            processed_rows.append(processed_row)

    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(processed_rows)


for i in range(14):
    input_file = f'day{i+1}.csv' 
    output_file = f'day{i+1}.csv' 
    remove_comments_from_csv(input_file, output_file)

for j in range(8, 15):
    input_file = f'poison{j}.csv' 
    output_file = f'poison{j}.csv'  
    remove_comments_from_csv(input_file, output_file)