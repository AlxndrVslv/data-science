#!/bin/bash

# Исключаемые директории
EXCLUDE_DIRS=".git venv .venv __pycache__ .idea .vscode"

# Функция проверки, нужно ли исключить путь
should_exclude() {
    local path="$1"
    for excl in $EXCLUDE_DIRS; do
        if [[ "$path" == "./$excl"* ]] || [[ "$path" == *"/$excl/"* ]]; then
            return 0  # исключить
        fi
    done
    return 1  # не исключать
}

# Переименование файлов .py и .ipynb
find . -type f \( -name "*.py" -o -name "*.ipynb" \) | while read file; do
    if should_exclude "$file"; then
        continue
    fi
    
    dir=$(dirname "$file")
    oldname=$(basename "$file")
    newname=$(echo "$oldname" | sed 's/ /_/g' | sed 's/\([a-z0-9]\)\([A-Z]\)/\1_\L\2/g' | tr '[:upper:]' '[:lower:]')
    
    if [ "$oldname" != "$newname" ]; then
        git mv "$dir/$oldname" "$dir/$newname" 2>/dev/null || mv "$dir/$oldname" "$dir/$newname"
        echo "Переименован: $oldname -> $newname"
    fi
done

# Переименование папок
find . -type d | while read dir; do
    if should_exclude "$dir"; then
        continue
    fi
    
    olddir=$(basename "$dir")
    newdir=$(echo "$olddir" | sed 's/ /_/g' | sed 's/\([a-z0-9]\)\([A-Z]\)/\1_\L\2/g' | tr '[:upper:]' '[:lower:]')
    
    if [ "$olddir" != "$newdir" ] && [ "$olddir" != "." ] && [ "$olddir" != ".." ]; then
        parent=$(dirname "$dir")
        git mv "$parent/$olddir" "$parent/$newdir" 2>/dev/null || mv "$parent/$olddir" "$parent/$newdir"
        echo "Переименована папка: $olddir -> $newdir"
    fi
done

echo "Готово! venv не был затронут."
