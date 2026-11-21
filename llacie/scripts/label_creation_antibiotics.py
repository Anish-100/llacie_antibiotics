import openpyxl
from pathlib import Path

SEPARATOR = '######################################'


def insert_into_workbook(all_notes, all_labels, output_path: Path):
    wb = openpyxl.Workbook()
    page = wb.active
    page.cell(row=1, column=1, value='ID_Number')
    page.cell(row=1, column=2, value='Labels')
    page.cell(row=1, column=3, value='Entire Note')

    for i, (note, label) in enumerate(zip(all_notes, all_labels), start=2):
        page.cell(row=i, column=1, value=i)
        page.cell(row=i, column=2, value=label)
        page.cell(row=i, column=3, value=note)

    wb.save(filename=output_path)


def get_all_notes(pathname: Path):
    notes_list = []
    with open(pathname, 'r') as f:
        current_lines = []
        for line in f:
            if line.strip() == SEPARATOR:
                if current_lines:
                    notes_list.append('\n'.join(current_lines))
                    current_lines = []
            else:
                current_lines.append(line.strip())
        if current_lines:
            notes_list.append('\n'.join(current_lines))
    return notes_list


def get_all_labels(pathname: Path):
    labels = []
    with open(pathname, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                if '. ' in line:
                    _, label = line.split('. ', 1)
                    labels.append(label)
                else:
                    labels.append(line)
    return labels


def main():
    all_notes = get_all_notes(Path('examples/antibiotics/synthetic_antibiotics_notes.txt'))
    all_labels = get_all_labels(Path('examples/antibiotics/synthetic_antibiotics_labels.txt'))
    output_path = Path('examples/antibiotics/synthetic_antibiotics_labels-labels.xlsx')
    insert_into_workbook(all_notes, all_labels, output_path)


if __name__ == '__main__':
    main()
