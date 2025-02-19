# Copyright (c) Meta Platforms, Inc. and affiliates.

_print() {
    local total_lines=$(awk 'END {print NR}' "$CURRENT_FILE")
    echo "[File: $(realpath "$CURRENT_FILE") ($total_lines lines total)]"
    lines_above=$(jq -n "$CURRENT_LINE - $WINDOW/2" | jq '[0, .] | max | floor')
    lines_below=$(jq -n "$total_lines - $CURRENT_LINE - $WINDOW/2" | jq '[0, .] | max | round')
    if [ $lines_above -gt 0 ]; then
        echo "($lines_above more lines above)"
    fi
    cat "$CURRENT_FILE" | grep -n $ | head -n $(jq -n "[$CURRENT_LINE + $WINDOW/2, $WINDOW/2] | max | floor") | tail -n $(jq -n "$WINDOW")
    if [ $lines_below -gt 0 ]; then
        echo "($lines_below more lines below)"
    fi
}

_constrain_line() {
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi
    local max_line=$(awk 'END {print NR}' "$CURRENT_FILE")
    local half_window=$(jq -n "$WINDOW/2" | jq 'floor')
    export CURRENT_LINE=$(jq -n "[$CURRENT_LINE, $max_line - $half_window] | min")
    export CURRENT_LINE=$(jq -n "[$CURRENT_LINE, $half_window] | max")
}

# @yaml
# signature: open "<path>" [<line_number>]
# docstring: opens the file at the given path in the editor. If line_number is provided, the window will be move to include that line
# arguments:
#   path:
#     type: string
#     description: the path to the file to open
#     required: true
#   line_number:
#     type: integer
#     description: the line number to move the window to (if not provided, the window will start at the top of the file)
#     required: false
open() {
    if [ -z "$1" ]
    then
        echo "Usage: open \"<file>\""
        return
    fi

    if [ -f "$1" ]; then
        if file -b --mime-type "$1" | grep -qE 'image'; then
            echo "Error: $1 is an image file. Current environment does not support image viewing."
            return
        fi
    elif [ -d "$1" ]; then
        echo "Error: $1 is a directory. You can only open files. Use cd or ls to navigate directories."
        return
    else
        echo "File $1 not found"
        return
    fi

    # Check if the second argument is provided
    if [ -n "$2" ]; then
        # Check if the provided argument is a valid number
        if ! [[ $2 =~ ^[0-9]+$ ]]; then
            echo "Usage: open \"<file>\" [<line_number>]"
            echo "Error: <line_number> must be a number"
            return  # Exit if the line number is not valid
        fi
        # Check if the file is larger than 20000 characters
        local char_count=$(wc -c < "$1")
        if [ "$char_count" -gt 20000 ]; then
            echo "The file is larger than 20000 characters, please use 'head -n NUM_LINES' to open it"
            return  # Exit if the file is too large
        fi
        local max_line=$(awk 'END {print NR}' $1)
        if [ $2 -gt $max_line ]; then
            echo "Warning: <line_number> ($2) is greater than the number of lines in the file ($max_line)"
            echo "Warning: Setting <line_number> to $max_line"
            local line_number=$(jq -n "$max_line")  # Set line number to max if greater than max
        elif [ $2 -lt 1 ]; then
            echo "Warning: <line_number> ($2) is less than 1"
            echo "Warning: Setting <line_number> to 1"
            local line_number=$(jq -n "1")  # Set line number to 1 if less than 1
        else
            local OFFSET=$(jq -n "$WINDOW/6" | jq 'floor')
            local line_number=$(jq -n "[$2 + $WINDOW/2 - $OFFSET, 1] | max | floor")
        fi
    else
        local line_number=$(jq -n "$WINDOW/2")  # Set default line number if not provided
    fi

    export CURRENT_FILE=$(realpath "$1")
    export CURRENT_LINE=$line_number
    _constrain_line
    _print
}

# @yaml
# signature: goto <line_number>
# docstring: moves the window to show <line_number>
# arguments:
#   line_number:
#     type: integer
#     description: the line number to move the window to
#     required: true
goto() {
    if [ $# -gt 1 ]; then
        echo "goto allows only one line number at a time."
        return
    fi
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi
    if [ -z "$1" ]
    then
        echo "Usage: goto <line>"
        return
    fi
    if ! [[ $1 =~ ^[0-9]+$ ]]
    then
        echo "Usage: goto <line>"
        echo "Error: <line> must be a number"
        return
    fi
    local max_line=$(awk 'END {print NR}' "$CURRENT_FILE")
    if [ $1 -gt $max_line ]
    then
        echo "Error: <line> must be less than or equal to $max_line"
        return
    fi
    local OFFSET=$(jq -n "$WINDOW/6" | jq 'floor')
    export CURRENT_LINE=$(jq -n "[$1 + $WINDOW/2 - $OFFSET, 1] | max | floor")
    _constrain_line
    _print
}

_scroll_warning_message() {
    # Warn the agent if we scroll too many times
    # Message will be shown if scroll is called more than WARN_AFTER_SCROLLING_TIMES (default 3) times
    # Initialize variable if it's not set
    export SCROLL_COUNT=${SCROLL_COUNT:-0}
    # Reset if the last command wasn't about scrolling
    if [ "$LAST_ACTION" != "scroll_up" ] && [ "$LAST_ACTION" != "scroll_down" ]; then
        export SCROLL_COUNT=0
    fi
    # Increment because we're definitely scrolling now
    export SCROLL_COUNT=$((SCROLL_COUNT + 1))
    if [ $SCROLL_COUNT -ge ${WARN_AFTER_SCROLLING_TIMES:-3} ]; then
        echo ""
        echo "WARNING: Scrolling many times in a row is very inefficient."
        echo "If you know what you are looking for, use \`search_file <pattern>\` instead."
        echo ""
    fi
}

# @yaml
# signature: scroll_down
# docstring: moves the window down {WINDOW} lines
scroll_down() {
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi
    export CURRENT_LINE=$(jq -n "$CURRENT_LINE + $WINDOW - $OVERLAP")
    _constrain_line
    _print
    _scroll_warning_message
}

# @yaml
# signature: scroll_up
# docstring: moves the window down {WINDOW} lines
scroll_up() {
    if [ -z "$CURRENT_FILE" ]
    then
        echo "No file open. Use the open command first."
        return
    fi
    export CURRENT_LINE=$(jq -n "$CURRENT_LINE - $WINDOW + $OVERLAP")
    _constrain_line
    _print
    _scroll_warning_message
}

# @yaml
# signature: create <filename>
# docstring: creates and opens a new file with the given name
# arguments:
#   filename:
#     type: string
#     description: the name of the file to create
#     required: true
create() {
    if [ -z "$1" ]; then
        echo "Usage: create <filename>"
        return
    fi

    # Check if the file already exists
    if [ -e "$1" ]; then
        echo "Error: File '$1' already exists."
		open "$1"
        return
    fi

    # Create the file with no content
    printf "\n" > "$1"
    # Use the existing open command to open the created file
    open "$1"
}
