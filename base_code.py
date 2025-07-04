# dq_email_decorators.py
import os
import smtplib
from email.message import EmailMessage
from functools import wraps
from typing import Callable, List, Union

from pyspark.sql import DataFrame, functions as F


# -------------------------------------------------------------------
# 1)  E-mail utility
# -------------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.example.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "noreply@example.com")
SMTP_PW   = os.getenv("SMTP_PW",   "********")          # never hard-code
ALERT_TO  = os.getenv("ALERT_TO",  "data-ops@example.com")

def send_alert(subject: str, body: str) -> None:
    """Minimal SMTP alert (STARTTLS)."""
    msg = EmailMessage()
    msg["From"] = SMTP_USER
    msg["To"]   = ALERT_TO
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
        smtp.starttls()
        smtp.login(SMTP_USER, SMTP_PW)
        smtp.send_message(msg)


# -------------------------------------------------------------------
# 2)  Decorators
# -------------------------------------------------------------------
def expect_row_count(expected: Union[int, Callable[[int], bool]]):
    """
    Decorator factory: verify row-count, e-mail if rule fails.
    `expected` can be an int (exact match) or a lambda returning bool.
    """
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs) -> DataFrame:
            df: DataFrame = func(*args, **kwargs)

            n = df.count()
            ok = (n == expected) if isinstance(expected, int) else expected(n)

            if not ok:
                send_alert(
                    subject=f"[DQ-FAIL] Row count check in {func.__name__}",
                    body=f"Expected {expected}, but got {n} rows.\n"
                         f"Spark app: {spark.sparkContext.applicationId}"
                )
            return df
        return _wrapper
    return _decorator


def assert_non_null(columns: List[str]):
    """
    Decorator factory: ensure given columns contain no NULLs; e-mail on failure.
    """
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs) -> DataFrame:
            df: DataFrame = func(*args, **kwargs)

            null_cond = F.lit(False)
            for c in columns:
                null_cond |= F.col(c).isNull()

            offending = df.filter(null_cond).limit(10)  # sample a few rows
            null_cnt  = offending.count()

            if null_cnt:
                sample_rows = "\n".join(map(str, offending.collect()))
                send_alert(
                    subject=f"[DQ-FAIL] NULLs detected in {func.__name__}",
                    body=(
                        f"{null_cnt} rows have NULLs in columns {columns}.\n\n"
                        f"Sample (first 10):\n{sample_rows}"
                    )
                )
            return df
        return _wrapper
    return _decorator