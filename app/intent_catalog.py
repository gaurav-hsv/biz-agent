INTENTS = [
  {
    "topic": "recommend_engagement",
    "required_fields": ["workload,incentive_type"],
    "filter_fields": ["name","workload","incentive_type"],
    "answer_fields": ["name","goal"]
  },
  {
    "topic": "customer_qualification",
    "required_fields": ["name | (workload,incentive_type)"],
    "filter_fields": ["name","workload","incentive_type"],
    "answer_fields": ["customer_qualification"]
  },
  {
    "topic": "partner_qualification",
    "required_fields": ["name | (workload,incentive_type)"],
    "filter_fields": ["name","workload","incentive_type"],
    "answer_fields": ["partner_qualification","solution_partner_designation","partner_specialization"]
  },
  {
    "topic": "earning_amount",
    "required_fields": ["name | (workload,incentive_type)","market","segment"],
    "filter_fields": ["name","workload","incentive_type"],
    "answer_fields": ["incentive_market_a","incentive_market_b","maximum_incentive_earning","enterprise_rate","smec_rate"]
  },
  {
    "topic": "activity_requirement",
    "required_fields": ["name | (workload,incentive_type)"],
    "filter_fields": ["name","workload","incentive_type"],
    "answer_fields": ["activity_requirement","min_hours"]
  }
]
