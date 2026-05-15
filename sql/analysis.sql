SELECT
  group,
  COUNT(*) as users,
  SUM(converted) as conversions,
  SUM(converted) / COUNT(*) as conversion_rate
FROM experiment_data
GROUP BY group;