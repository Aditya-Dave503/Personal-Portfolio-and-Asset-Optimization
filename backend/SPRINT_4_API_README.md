# Holdings and portfolio API

All routes below are under `/api/v1` and require the bearer token returned by
`POST /auth/login`. The authenticated user ID is passed to the service layer;
clients cannot submit an owner ID.

## Holdings

| Method | Path | Success |
| --- | --- | --- |
| POST | `/holdings/` | `201`, holding object |
| GET | `/holdings/` | `200`, array of holding objects |
| PUT | `/holdings/{holding_id}` | `200`, updated holding object |
| DELETE | `/holdings/{holding_id}` | `204` |

Create request:

```json
{
  "asset_type": "NIFTY_EQUITIES",
  "name": "Nifty 50 ETF",
  "units": "4.5",
  "invested_value": "12000.00"
}
```

`asset_type` accepts `STOCK`, `MF`, `GOLD`, `SILVER`, and `BOND`, matching the
project's current holdings vocabulary. Units and invested value must be
positive. Silver holdings can be recorded even though silver is excluded from
the optimizer's asset universe.

## Portfolios

| Method | Path | Success |
| --- | --- | --- |
| POST | `/portfolios/optimize` | `201`, generated portfolio |
| GET | `/portfolios/` | `200`, array of the user's portfolios |
| GET | `/portfolios/{portfolio_id}` | `200`, portfolio |
| GET | `/portfolios/{portfolio_id}/allocations` | `200`, allocation array |

Optimization request:

```json
{
  "investment_amount": "100000.00",
  "optimization_mode": "max_sharpe"
}
```

Allowed modes are `black_litterman`, `max_sharpe`, and `min_variance`. The
investment amount must be positive. Missing service modules respond with `503`
until the portfolio service implementation is integrated. Service `LookupError`
results map to `404`; optimizer `ValueError` results map to `422`. Unexpected
service failures remain server errors and are never converted into a success.

## Service integration contract

The endpoints call `services.holding_service` functions named `create_holding`,
`list_holdings`, `update_holding`, and `delete_holding`, with the agreed
`(db, user_id, ...)` arguments. Portfolio endpoints call
`services.portfolio_service` functions `optimize_portfolio`, `list_portfolios`,
`get_portfolio`, and `get_portfolio_allocations`, also with `db` and authenticated
`user_id`. These service modules are not present in this checkout yet; the API
does not add persistence or ML logic in their place.
