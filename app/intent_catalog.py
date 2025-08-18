INTENTS = [
  {
    "topic": "recommend_engagement",
    "required_fields": ["name | (workload,incentive_type)"],
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
    "required_fields": ["name | (workload,incentive_type)"],
    "filter_fields": ["name","workload","incentive_type"],
    "answer_fields": ["incentive_market_a","incentive_market_b","maximum_incentive_earning","enterprise_rate","smec_rate"]
  },
  {
    "topic": "calc_presales_workshop_payout",
    "required_fields": [
      "name | (workload,incentive_type)", 
      "country",                           
      "acv",                              
      "hours"                              
    ],
    "filter_fields": ["name","workload","incentive_type"],
    "answer_fields": [
      # Needed to compute + explain branches
      "incentive_market_a","incentive_market_b","incentive_market_c",
      "maximum_incentive_earning",
      "workshop_rate_hourly_a","workshop_rate_hourly_b","workshop_rate_hourly_c"
    ]
  },
  {
    "topic": "calc_presales_briefing_payout",
    "required_fields": [
      "name | (workload,incentive_type)",
      "country" 
    ],
    "filter_fields": ["name","workload","incentive_type"],
    "answer_fields": [
      "incentive_market_a","incentive_market_b","incentive_market_c"
    ]
  },

  {
    "topic": "activity_requirement",
    "required_fields": ["name | (workload,incentive_type)"],
    "filter_fields": ["name","workload","incentive_type"],
    "answer_fields": ["activity_requirement","min_hours"]
  }
]
