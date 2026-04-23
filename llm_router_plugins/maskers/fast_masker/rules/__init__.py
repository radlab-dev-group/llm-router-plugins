from typing import Optional, Callable

"""
Package that contains concrete masking rule implementations.
"""

from .credit_card_rule import CreditCardRule
from .nrb_rule import NrbRule
from .vin_rule import VinRule
from .mac_address_rule import MacAddressRule
from .jwt_rule import JwtRule
from .sim_card_rule import SimCardRule
from .ssl_cert_rule import SslCertRule
from .transaction_ref_rule import TransactionRefRule
from .invoice_number_rule import InvoiceNumberRule
from .order_number_rule import OrderNumberRule
from .car_plate_rule import CarPlateRule
from .health_id_rule import HealthIdRule
from .id_card_rule import IdCardRule
from .passport_rule import PassportRule
from .pesel_tagged_rule import PeselTaggedRule
from .pesel_rule import PeselRule
from .phone_international_rule import PhoneInternationalRule
from .phone_rule import PhoneRule
from .email_rule import EmailRule
from .url_rule import UrlRule
from .ip_rule import IpRule
from .bank_account_rule import BankAccountRule
from .date_number_rule import DateNumberRule
from .date_word_rule import DateWordRule
from .krs_rule import KrsRule
from .money_rule import MoneyRule
from .nip_rule import NipRule
from .postal_code_rule import PostalCodeRule
from .regon_rule import RegonRule
from .social_id_rule import SocialIdRule
from .ssn_rule import SsnRule
from .eu_vat_rule import EuVatRule
