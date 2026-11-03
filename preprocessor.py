import re
import pandas as pd

def preprocessor(data):
    # More flexible regex pattern to handle various datetime formats
    # Matches: 28/01/25, 6:45 pm or 28/01/2025, 6:45 PM or with hyphens, etc.
    pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4},\s\d{1,2}:\d{2}\s?(?:am|pm|AM|PM))\s-\s(.+)'
    matches = re.findall(pattern, data, re.IGNORECASE)
    
    if not matches:
        # Try alternative formats that don't require explicit am/pm
        pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4},\s\d{1,2}:\d{2})\s-\s(.+)'
        matches = re.findall(pattern, data)

    # Clean non-breaking spaces
    cleaned_matches = [(t.replace('\u202f', ' '), msg) for t, msg in matches]
    df = pd.DataFrame(cleaned_matches, columns=['date_time', 'Message'])

    users, messages = [], []
    user_pattern = r'^[\d\s()+-]+$'

    for message in df['Message']:
        parts = message.split(': ', 1)

        if len(parts) == 2:
            user_candidate, msg_text = parts[0].strip(), parts[1].strip()
            if re.match(user_pattern, user_candidate) or user_candidate:
                users.append(user_candidate)
                messages.append(msg_text)
            else:
                users.append('group_notification')
                messages.append(message.strip())
        else:
            users.append('group_notification')
            messages.append(message.strip())

    df['user'] = users
    df['message'] = messages
    
    # Convert date_time with flexible format handling
    def parse_datetime(dt_str):
        """Parse datetime with multiple format fallbacks."""
        dt_str = dt_str.replace('\u202f', ' ').strip()
        
        # Normalize am/pm to lowercase for consistency
        dt_str_normalized = re.sub(r'\s*(am|pm|AM|PM)\s*$', lambda m: ' ' + m.group(1).lower(), dt_str)
        
        formats = [
            '%d/%m/%Y, %I:%M %p',      # 28/01/2025, 6:45 pm
            '%d/%m/%y, %I:%M %p',      # 28/01/25, 6:45 pm
            '%d-%m-%Y, %I:%M %p',      # 28-01-2025, 6:45 pm
            '%d-%m-%y, %I:%M %p',      # 28-01-25, 6:45 pm
            '%d/%m/%Y, %H:%M',         # 28/01/2025, 18:45
            '%d/%m/%y, %H:%M',         # 28/01/25, 18:45
            '%d-%m-%Y, %H:%M',         # 28-01-2025, 18:45
            '%d-%m-%y, %H:%M',         # 28-01-25, 18:45
            '%d/%m/%Y, %I:%M%p',       # 28/01/2025, 6:45pm (no space)
            '%d/%m/%y, %I:%M%p',       # 28/01/25, 6:45pm (no space)
        ]
        
        for fmt in formats:
            try:
                return pd.to_datetime(dt_str_normalized, format=fmt)
            except (ValueError, TypeError):
                continue
        
        # Last resort: infer format
        try:
            return pd.to_datetime(dt_str_normalized, dayfirst=True, infer_datetime_format=True)
        except:
            return None
    
    df['date_time'] = df['date_time'].apply(parse_datetime)
    
    # Drop rows with parsing errors
    df = df.dropna(subset=['date_time'])
    
    if len(df) == 0:
        # Debug: Show what was extracted
        import sys
        print(f"ERROR: No valid datetime entries found!", file=sys.stderr)
        if not matches:
            print(f"ERROR: Regex didn't match any entries. Check your chat format.", file=sys.stderr)
        else:
            print(f"ERROR: Regex found {len(matches)} entries but datetime parsing failed.", file=sys.stderr)
            print(f"Sample extracted datetime strings: {[m[0] for m in matches[:3]]}", file=sys.stderr)
        raise ValueError("No valid datetime entries found. Check the chat format. Expected format: DD/MM/YYYY, HH:MM am/pm - User: Message")

    df['message_id'] = [f"msg_{index:06d}" for index in range(1, len(df) + 1)]
    df['date'] = df['date_time'].dt.date
    df['date'] = pd.to_datetime(df['date'])
    df['time'] = df['date_time'].dt.time
    df['year'] = df['date_time'].dt.year
    df['month_num'] = df['date_time'].dt.month
    df['month'] = df['date_time'].dt.month_name()
    df['day'] = df['date_time'].dt.day
    df['hour'] = df['date_time'].dt.hour
    df['minute'] = df['date_time'].dt.minute
    df['day_name'] = df['date_time'].dt.day_name()

    # Create period bins
    df['period'] = df['hour'].apply(lambda h: f"{h:02d}-{'00' if h == 23 else f'{(h + 1)%24:02d}' }")

    return df